"""Streamlit UI running on Laptop 2 for evaluator demonstration."""

import json
import requests
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="PathSnap Evaluator UI",
    page_icon="🗺️",
    layout="wide",
)

st.title("🗺️ PathSnap: Distributed Map-Matching & Route Reconstruction")
st.markdown(
    """
    **Distributed Map-Matching & Route Reconstruction** | Two-Laptop Evaluation Architecture
    - **Laptop 1 (Backend Server)**: Hosts FastAPI, trained ML models, HMM/Viterbi decoder, and ground-truth validation.
    - **Laptop 2 (Evaluator UI)**: Queries the API and renders real-time route reconstructions with performance metrics.
    """
)

# Sidebar Configuration
st.sidebar.header("⚙️ Laptop 1 Connection")
api_base_url = st.sidebar.text_input("FastAPI Server URL", value="http://localhost:8000")

# Test connection
server_healthy = False
try:
    health_resp = requests.get(f"{api_base_url}/health", timeout=3)
    if health_resp.status_code == 200:
        server_healthy = True
        st.sidebar.success(" Connected to Laptop 1 Server")
    else:
        st.sidebar.warning(f"⚠️ Server returned status {health_resp.status_code}")
except Exception:
    st.sidebar.error("❌ Cannot reach Laptop 1 Server. Check IP and port.")

st.sidebar.header("📋 Evaluation Trip Selection")
partition = st.sidebar.selectbox("Partition", options=["student_test", "evaluator_test"], index=0)

# Fetch available trips from server
available_trips = []
if server_healthy:
    try:
        trips_resp = requests.get(f"{api_base_url}/trips", timeout=3)
        if trips_resp.status_code == 200:
            available_trips = trips_resp.json().get(partition, [])
    except Exception:
        pass

if not available_trips:
    # Fallback default IDs for demonstration
    available_trips = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000083", "00000092",
    ]

selected_trip = st.sidebar.selectbox("Trip ID", options=available_trips)
decoder_mode = st.sidebar.selectbox(
    "Decoding Strategy",
    options=["ml_hmm", "distance_baseline", "classifier_only"],
    format_func=lambda x: {
        "ml_hmm": "ML Probability + HMM Viterbi (Proposed)",
        "distance_baseline": "Distance-only Baseline HMM",
        "classifier_only": "Classifier-only Argmax",
    }.get(x, x),
)

run_button = st.sidebar.button("🚀 Reconstruct Route & Evaluate", type="primary")

if run_button:
    with st.spinner(f"Reconstructing route for trip {selected_trip} via Laptop 1 API..."):
        try:
            url = f"{api_base_url}/evaluate/{selected_trip}?partition={partition}&mode={decoder_mode}"
            resp = requests.get(url, timeout=30)

            if resp.status_code != 200:
                st.error(f"Error from API ({resp.status_code}): {resp.text}")
            else:
                data = resp.json()
                metrics = data.get("metrics", {})

                st.subheader(f"📊 Evaluation Metrics for Trip: {selected_trip}")

                # Metrics cards
                col1, col2, col3, col4, col5, col6 = st.columns(6)

                with col1:
                    top1 = metrics.get("top1_accuracy")
                    st.metric(
                        "Top-1 Accuracy",
                        f"{top1 * 100:.2f}%" if top1 is not None else "N/A",
                    )
                with col2:
                    f1 = metrics.get("macro_f1")
                    st.metric(
                        "Macro F1",
                        f"{f1:.4f}" if f1 is not None else "N/A",
                    )
                with col3:
                    cont = metrics.get("route_continuity")
                    st.metric(
                        "Route Continuity",
                        f"{cont:.1f}%" if cont is not None else "N/A",
                    )
                with col4:
                    miss = metrics.get("candidate_miss_rate")
                    st.metric(
                        "Miss Rate",
                        f"{miss * 100:.2f}%" if miss is not None else "N/A",
                    )
                with col5:
                    ll = metrics.get("log_loss")
                    st.metric(
                        "Log Loss",
                        f"{ll:.4f}" if ll is not None else "N/A",
                    )
                with col6:
                    t_inf = metrics.get("inference_time_sec")
                    st.metric(
                        "Inference Time",
                        f"{t_inf:.3f} s" if t_inf is not None else "N/A",
                    )

                # Reconstructed Route Summary
                st.markdown("---")
                st.subheader("🗺️ Interactive Route Map")
                st.caption(
                    "Blue: Raw GPS Points | Green: Ground-Truth Traversed Route (.route) | Red: PathSnap Predicted Route"
                )

                map_html = data.get("folium_map_html", "")
                if map_html:
                    components.html(map_html, height=550, scrolling=True)
                else:
                    st.info("No map data returned.")

                # Trajectory details
                st.markdown("---")
                with st.expander("🔍 View Sequence Details"):
                    st.write(f"**Total GPS Observations:** {data.get('num_gps_points')}")
                    st.write(f"**Reconstructed Arc Sequence ({len(data.get('reconstructed_route', []))} segments):**")
                    st.code(str(data.get("reconstructed_route", [])))

        except Exception as e:
            st.error(f"Failed to communicate with Laptop 1 server: {str(e)}")
else:
    st.info("👈 Select a trip in the sidebar and click **'Reconstruct Route & Evaluate'** to run the evaluation.")
