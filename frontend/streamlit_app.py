import streamlit as st
import requests
import pandas as pd
from bs4 import BeautifulSoup
import os

st.title("SHL Assessment Recommender")

st.write("Enter a job description or provide a URL to get the most relevant SHL assessments.")

# Create tabs for input methods
job_description_tab, url_tab = st.tabs(["Job Description", "Job URL"])

job_description = ""
url = ""

with job_description_tab:
    job_description = st.text_area("Job Description:")

with url_tab:
    url = st.text_input("Job Description URL:")
    if url:
        try:
            page = requests.get(url, timeout=30)
            page.raise_for_status()
            soup = BeautifulSoup(page.content, 'html.parser')
            job_description = soup.get_text().strip()  # Directly use the parsed job description
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to fetch the URL provided: {e}")
        except Exception as e:
            st.error("Failed to parse job description from the provided URL. Please ensure the URL points to an accessible page.")

# Read backend URL from environment variable
backend_url = os.environ.get("BACKEND_API_URL")
if not backend_url:
    st.error("Configuration error: BACKEND_API_URL is not set. Set the BACKEND_API_URL environment variable to your backend base URL (e.g., https://api.example.com) and reload the app.")

if st.button("Get Recommendations", disabled=(backend_url is None)):
    # Proceed if either job_description is filled or a URL is provided and parsed
    if not job_description.strip() and not url.strip():
        st.error("Please enter a job description or provide a valid URL.")
    else:
        with st.spinner('Analyzing job description and fetching recommendations...'):
            # Normalize backend url and construct endpoint
            backend_url = backend_url.rstrip('/')
            recommend_endpoint = f"{backend_url}/recommend"

            try:
                # Add timeout to prevent hanging
                response = requests.post(
                    recommend_endpoint,
                    json={"job_description": job_description},
                    timeout=300  # 2 minute timeout since LLM processing can take time
                )

                # Raise for non-2xx responses to handle gracefully
                if response.status_code != 200:
                    # Try to extract message from response
                    try:
                        err = response.json().get('detail') or response.text
                    except Exception:
                        err = response.text
                    st.error(f"Error from recommendation service: \n{response.text}")
                else:
                    try:
                        response_json = response.json()
                    except ValueError:
                        st.error("Invalid response from recommendation service. Please try again later.")
                        response_json = {}

                    # API returns 'recommended_assessments' per spec
                    recommendations = response_json.get("recommended_assessments", [])

                    if recommendations:
                        st.success(f"Found {len(recommendations)} relevant SHL assessments!")

                        # Convert to DataFrame for display
                        df = pd.DataFrame(recommendations)

                        # Assign correct column names to match backend schema
                        df = df.rename(columns={
                            "name": "Assessment Name",
                            "url": "URL",
                            "remote_support": "Remote Support",
                            "adaptive_support": "Adaptive/IRT Support",
                            "duration": "Duration (mins)",
                            "test_type": "Test Types"
                        })

                        # Format test_type as comma-separated string if it's a list
                        if 'Test Types' in df.columns:
                            df['Test Types'] = df['Test Types'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)

                        # Ensure duration is an integer column
                        if 'Duration (mins)' in df.columns:
                            df['Duration (mins)'] = df['Duration (mins)'].fillna(0).astype(int)

                        # Display the recommendations in a table
                        st.dataframe(df.reset_index(drop=True))

                        # Provide click-through links below the table (clean, accessible)
                        if 'URL' in df.columns:
                            st.markdown("**Assessment links:**")
                            for idx, row in df.iterrows():
                                url_link = row.get('URL')
                                name = row.get('Assessment Name') or url_link
                                if url_link:
                                    st.markdown(f"- [{name}]({url_link})")

                        # Add a download button for CSV export
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download recommendations as CSV",
                            data=csv,
                            file_name="shl_recommendations.csv",
                            mime="text/csv",
                        )
                    else:
                        st.warning("No matching assessments found for this job description. Try providing more details about the role and required skills.")
            except requests.exceptions.ConnectionError:
                st.error(f"Connection error: Could not reach the recommendation service at {backend_url}. Please ensure the BACKEND_API_URL is correct and the backend is running.")
            except requests.exceptions.Timeout:
                st.error("The recommendation service is taking too long to respond. Please try again later.")
            except requests.exceptions.RequestException as e:
                st.error(f"Network error while contacting the recommendation service: {e}")
            except Exception as e:
                st.error(f"An unexpected error occurred. Please try again or contact support.")

