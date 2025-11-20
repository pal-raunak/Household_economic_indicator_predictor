# .\venv\Scripts\Activate.ps1
# streamlit run app/app.py

import streamlit as st
from utils import io as custom_io, preprocess, segmentation, prediction
# import yaml  # Not needed when LLM is commented out
# from utils.azureopenai import UtilsAzureOpenAI
import os
import pandas as pd
import matplotlib.pyplot as plt

# llm = UtilsAzureOpenAI().get_llm()


def get_actual_data(tiff_filename, id_value):

    try:
        # Get the base name without extension
        base_name = os.path.splitext(os.path.basename(tiff_filename))[0]
        excel_path = os.path.join('app', 'static', 'data', f'{base_name}.xlsx')

        try:

            with open(excel_path, 'rb') as f:
                df = pd.read_excel(f)

            # Convert ID column to string for comparison
            df['id'] = df['id'].astype(str)
            id_value = str(id_value)

            # Filter for the specific ID``
            actual_data = df[df['id'] == id_value]
            if not actual_data.empty:
                return actual_data.iloc[0].to_dict()
            else:
                st.warning(f"ID {id_value} not found in Excel file")
                return None

        except PermissionError:
            st.error(
                f"Permission denied: Cannot access the Excel file at {excel_path}")
            st.info(
                "Please ensure the Excel file is not open in another program and try again.")
            return None
        except FileNotFoundError:
            st.error(f"Excel file not found at: {excel_path}")
            return None

    except Exception as e:
        st.error(f"Error loading actual data: {str(e)}")
        return None


# Try to import streamlit-extras for colored headers
try:
    from streamlit_extras.colored_header import colored_header
    USE_COLORED_HEADER = True
except ImportError:
    USE_COLORED_HEADER = False

# Map streamlit-extras color names to CSS color names for fallback
COLOR_MAP = {
    'blue-70': 'blue',
    'orange-70': 'orange',
    'green-70': 'green',
    'cyan-70': 'cyan',
}


def section_header(label, color='green-70'):
    if USE_COLORED_HEADER:
        colored_header(label, description=None, color_name=color)
    else:
        css_color = COLOR_MAP.get(color, color)
        st.markdown(
            f"<h2 style='color:{css_color};margin-bottom:0.5em'>{label}</h2>", unsafe_allow_html=True)


st.set_page_config(page_title="Economic Indicator Predictor", layout="wide")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        max-width: 1200px;
        margin: 0 auto;
    }
    .stButton>button {
        width: 100%;
        margin: 0 auto;
        display: block;
    }
    .prediction-box {
        border: 2px solid #4CAF50;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .mask-container {
        text-align: center;
        padding: 10px 0 0 0;
        border: 1px solid #ddd;
        border-radius: 5px;
        margin: 5px 0 20px 0;
        background: #18181811;
    }
    .mask-container img {
        max-width: 250px;
        height: auto;
        display: block;
        margin: 0 auto;
    }
    .stSelectbox {
        max-width: 200px;
        margin: 0 auto;
    }
    .stImage {
        display: block;
        margin: 0 auto;
    }
    .element-container {
        margin: 0;
        padding: 0;
    }
    .stSuccess {
        display: none;
    }
    .reset-btn {
        margin-bottom: 1.5em;
    }
    .big-reset-btn {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100vw;
        margin: 2.5em 0 2em 0;
    }
    .big-reset-btn button {
        font-size: 3.2em !important;
        padding: 2.5em 4em !important;
        background-color: #ff5252 !important;
        color: white !important;
        border-radius: 24px !important;
        font-weight: bold !important;
        letter-spacing: 0.07em;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.18);
        width: auto !important;
        min-width: 600px !important;
        max-width: 90vw !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .big-reset-btn-row {
        display: flex;
        justify-content: center;
        align-items: center;
        width: 100vw;
        margin: 2.5em 0 2em 0;
        gap: 2em;
    }
    .big-action-btn button {
        font-size: 3.2em !important;
        padding: 2.5em 4em !important;
        color: white !important;
        border-radius: 24px !important;
        font-weight: bold !important;
        letter-spacing: 0.07em;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.18);
        width: auto !important;
        min-width: 300px !important;
        max-width: 90vw !important;
        margin: 0 auto !important;
        display: block !important;
    }
    .reset-btn button {
        background-color: #ff5252 !important;
    }
    .download-btn button {
        background-color: #4CAF50 !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Household Level Economic Indicator Predictor")

# Initialize session state
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'cropped_img_path' not in st.session_state:
    st.session_state.cropped_img_path = None
if 'pixel' not in st.session_state:
    st.session_state.pixel = None
if 'masks' not in st.session_state:
    st.session_state.masks = None
if 'annotated_images' not in st.session_state:
    st.session_state.annotated_images = None
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_action' not in st.session_state:
    st.session_state.current_action = None
if 'tiff_file' not in st.session_state:
    st.session_state.tiff_file = None
if 'excel_file' not in st.session_state:
    st.session_state.excel_file = None

# --- Sidebar Activity Log ---
with st.sidebar:
    # New Prediction button at the top
    if st.button("🔄 New Prediction", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

    st.markdown("---")
    st.header("📋 Activity Log")

    action = st.session_state.get("current_action")
    if action:
        action_type = action.get("type")

        if action_type == 'ID Selected':
            st.markdown(f"**Selected ID:** {action.get('id')}")

        elif action_type == 'House Selected':
            st.markdown(f"**Selected ID:** {action.get('id')}")
            st.markdown("House image:")
            st.image(action.get('image'), width=180)

        elif action_type == 'Mask Selected':
            st.markdown(f"**Selected ID:** {action.get('id')}")
            st.markdown("Working House image:")
            st.image(action.get('house_image'), width=180)
            st.markdown("Selected Mask:")
            st.image(action.get('mask_image'), width=180)

        elif action_type == 'Prediction Completed':
            st.markdown(f"**Prediction completed for ID:** {action.get('id')}")
            st.markdown(action.get('summary'))

        st.markdown("---")

    # Download button at the bottom of sidebar
    if 'predictions' in st.session_state and st.session_state.predictions:

        # Generate downloadable data
        predictions_data = st.session_state.get('predictions', {})
        if predictions_data:
            # Create a zip file containing both the text and image
            import io
            import zipfile
            from PIL import Image

            # Create text content
            prediction_output = f"Prediction Results:\n\n"
            prediction_output += "Rooftop Analysis:\n"
            prediction_output += f"Predicted Rooftop Type: {predictions_data['predictions']['rooftop_type'].title()}\n\n"
            prediction_output += "House Characteristics:\n"
            prediction_output += f"Floor Type: {predictions_data['predictions']['Floor'].title()}\n"
            prediction_output += f"Wall Type: {predictions_data['predictions']['Wall'].title()}\n"
            prediction_output += f"Water Supply: {predictions_data['predictions']['S.D.Water'].title()}\n"
            prediction_output += f"Govt. House Scheme: {predictions_data['predictions']['Govt.H.Sch'].title()}\n\n"
            prediction_output += "Social Indicators:\n"
            prediction_output += f"Occupation: {predictions_data['predictions']['Occupation'].title()}\n"
            prediction_output += f"Ration Card: {predictions_data['predictions']['Rationcard'].title()}\n"
            prediction_output += f"MGNREGA: {predictions_data['predictions']['MGNREGA'].title()}\n\n"
            prediction_output += "Vehicle Ownership:\n"
            prediction_output += f"Bike: {int(predictions_data['predictions']['Bike'])}\n"
            prediction_output += f"Cycle: {int(predictions_data['predictions']['cycle'])}\n"
            prediction_output += f"Car: {int(predictions_data['predictions']['car'])}\n"

            # Create a zip file in memory
            zip_buffer = io.BytesIO()
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add text file
                zip_file.writestr('prediction_results.txt', prediction_output)

                # Add both images
                if st.session_state.cropped_img_path:
                    # Add cropped house image
                    img = Image.open(st.session_state.cropped_img_path)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')
                    img_byte_arr.seek(0)
                    zip_file.writestr('cropped_house_area.png',
                                      img_byte_arr.getvalue())

                if st.session_state.box_img_path:
                    # Add image with bounding box
                    box_img = Image.open(st.session_state.box_img_path)
                    box_img_byte_arr = io.BytesIO()
                    box_img.save(box_img_byte_arr, format='PNG')
                    box_img_byte_arr.seek(0)
                    zip_file.writestr('output_with_box.png',
                                      box_img_byte_arr.getvalue())

            # Prepare the zip file for download
            zip_buffer.seek(0)
            st.download_button(
                "⬇️ Download Results (ZIP)",
                data=zip_buffer,
                file_name="prediction_results.zip",
                mime="application/zip",
                use_container_width=True
            )

# --- Step 1: TIFF and Excel Input ---
section_header(
    "Provide your Map (.tif) and Geo-Coordinates (.xlsx or .csv)", color="blue-70")

col1, col2 = st.columns([1, 1])

# Map Upload
with col1:
    left_align = st.container()
    with left_align:
        st.markdown("<div style='text-align: left;'>", unsafe_allow_html=True)
        tiff_choice = st.radio(
            "Map Options:",
            ["Use stored map", "Upload your own"],
            horizontal=True
        )
        if tiff_choice == "Upload your own":
            uploaded_tiff = st.file_uploader(
                "Upload Map", type=["tif", "tiff"])
            if uploaded_tiff:
                st.session_state.tiff_file = uploaded_tiff
        else:
            stored_files = custom_io.get_stored_tiffs()
            selected_tiff = st.selectbox("Choose Map", stored_files)
            if selected_tiff:
                st.session_state.tiff_file = selected_tiff
        st.markdown("</div>", unsafe_allow_html=True)

# Excel Upload
with col2:
    uploaded_excel = st.file_uploader(
        "Upload Geo-coordinates:", type=["xlsx", "csv"])
    if uploaded_excel:
        st.session_state.excel_file = uploaded_excel

# Proceed if both files are uploaded
if st.session_state.tiff_file and st.session_state.excel_file:
    try:
        with st.spinner("Loading Excel file..."):
            df = custom_io.load_excel(st.session_state.excel_file)

        st.dataframe(df, height=200)

        # Select ID
        col1, col2, col3 = st.columns([3, 1.5, 3])
        with col2:
            selected_id = st.selectbox("Select ID", df['id'])

            # Set action if not already set or is not House/Mask
            if (
                'current_action' not in st.session_state
                or st.session_state.current_action is None
                or st.session_state.current_action.get('type') not in ['House Selected', 'Mask Selected']
            ):
                st.session_state.current_action = {
                    'type': 'ID Selected',
                    'id': selected_id
                }

        # Process Selected ID
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Process Selected ID", use_container_width=True):
                try:
                    with st.spinner("Processing..."):
                        cropped_img_path, box_img_path, pixel = preprocess.crop_and_locate(
                            selected_id, df, st.session_state.tiff_file
                        )

                    st.session_state.cropped_img_path = cropped_img_path
                    st.session_state.box_img_path = box_img_path
                    st.session_state.pixel = pixel
                    st.session_state.current_action = {
                        'type': 'House Selected',
                        'id': selected_id,
                        'image': cropped_img_path,
                        'box_image': box_img_path
                    }
                    st.session_state.step = 2
                    st.rerun()

                except Exception as e:
                    st.error(f"Error processing image: {str(e)}")

    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")

# Add this function at the top of the file, after the imports


def get_image_base64(image_path):
    import base64
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode()


# --- Step 2: Show Cropped Image ---
if st.session_state.cropped_img_path:
    section_header("Here is your selected House:", color="orange-70")

    # Create two columns for images
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{get_image_base64(st.session_state.box_img_path)}' 
                 style='width: 700px; height: 500px; object-fit: contain;'>
        </div>
        """, unsafe_allow_html=True)

    with img_col2:
        st.markdown(f"""
        <div style='text-align: center;'>
            <img src='data:image/png;base64,{get_image_base64(st.session_state.cropped_img_path)}' 
                 style='width: 450px; height: 450px; object-fit: contain;'>
        </div>
        """, unsafe_allow_html=True)

    # Create a centered container for the button
    st.markdown("<br>", unsafe_allow_html=True)  # Add some vertical space
    button_col1, button_col2, button_col3 = st.columns([1, 2, 1])

    with button_col2:
        if st.button("Segment Now", use_container_width=True):
            try:
                with st.spinner("Segmenting..."):
                    masks, annotated_images = segmentation.generate_masks(
                        st.session_state.cropped_img_path,
                        st.session_state.pixel
                    )
                st.session_state.masks = masks
                st.session_state.annotated_images = annotated_images
                st.session_state.step = 3
                st.rerun()
            except Exception as e:
                st.error(f"Error in segmentation: {str(e)}")

# --- Step 3: Show Masks ---
if st.session_state.annotated_images:
    # st.header("Choose the best mask for your house:")
    section_header("Choose the best mask for your house:", color="blue-70")

    # Create three columns for masks with equal width
    mask_cols = st.columns([1, 1, 1])

    # Initialize selected mask in session state if not exists
    if 'selected_mask' not in st.session_state:
        st.session_state.selected_mask = None

    # Display masks with their respective buttons
    for idx, (col, img) in enumerate(zip(mask_cols, st.session_state.annotated_images)):
        with col:
            st.markdown('<div class="mask-container">', unsafe_allow_html=True)
            st.image(
                img, caption=f"Mask {idx + 1} (Point Prompted)", width=250, use_container_width=True)
            if st.button(f"Select Mask {idx + 1}", key=f"mask_btn_{idx}", use_container_width=True):
                st.session_state.selected_mask = idx
                st.session_state.current_action = {
                    'type': 'Mask Selected',
                    'id': selected_id,
                    'house_image': st.session_state.cropped_img_path,
                    'mask_image': st.session_state.annotated_images[idx]
                }
                st.success(f"Mask {idx + 1} selected!")
            st.markdown('</div>', unsafe_allow_html=True)

    # Show prediction button only if a mask is selected
    if st.session_state.selected_mask is not None:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("Run Predictor", use_container_width=True):
                try:
                    predictions = prediction.predict_from_mask(
                        st.session_state.masks[st.session_state.selected_mask])

                    # Add to history
                    st.session_state.history.append({
                        'id': selected_id,
                        'house_image': st.session_state.cropped_img_path,
                        'mask_image': st.session_state.annotated_images[st.session_state.selected_mask]
                    })

                    # Store predictions in session state for download
                    st.session_state.predictions = predictions
                    st.rerun()  # Rerun to show the prediction results
                except Exception as e:
                    st.error(f"Error in prediction: {str(e)}")

# --- Prediction Results Section ---
if 'predictions' in st.session_state and st.session_state.predictions:
    st.markdown('''
    <style>
    /* Individual prediction card styling */
    .pred-card { 
        background: #232323; 
        border-radius: 12px; 
        padding: 16px; 
        box-shadow: 0 2px 8px #0002; 
        transition: box-shadow 0.2s, background 0.2s; 
        height: 100%; 
        display: flex; 
        flex-direction: column;
        font-size: 0.9em;
    }
    .pred-card:hover { background: #333; box-shadow: 0 4px 16px #0004; }
    .pred-card h4 { margin-top: 0; font-size: 1.1em; }
    .pred-card ul { margin: 8px 0; padding-left: 20px; }
    .pred-card li { margin: 4px 0; }
    .actual-value { color: #4CAF50; font-weight: bold; }
    .predicted-value { color: #FFA726; font-weight: bold; }
    </style>
    ''', unsafe_allow_html=True)

    section_header("📊 Prediction Results:", color="green-70")

    predictions = st.session_state.predictions

    # Add LLM Summary right after the header
    # COMMENTED OUT: LLM Summary generation
    # summary_prompt = f"""Based on the following economic indicators, provide a 3-4 line summary of the household's economic well-being:
    # - Rooftop Type: {predictions['predictions']['rooftop_type']}
    # - Floor Type: {predictions['predictions']['Floor']}
    # - Wall Type: {predictions['predictions']['Wall']}
    # - Water Supply: {predictions['predictions']['S.D.Water']}
    # - Govt. House Scheme: {predictions['predictions']['Govt.H.Sch']}
    # - Occupation: {predictions['predictions']['Occupation']}
    # - Ration Card: {predictions['predictions']['Rationcard']}
    # - MGNREGA: {predictions['predictions']['MGNREGA']}
    # - Vehicles: {int(predictions['predictions']['Bike'])} bikes, {int(predictions['predictions']['cycle'])} cycles, {int(predictions['predictions']['car'])} cars
    # 
    # if "Govt. House Scheme" comes out to be "Pmay-N" that means housing scheme has nto been provided.  Provide a concise summary focusing on the economic status and living conditions."""

    # try:
    #     with st.spinner("Generating summary..."):
    #         summary_response = llm.invoke(summary_prompt)
    #         st.markdown(f"""
    #         <div style='margin: 20px 0;'>
    #             <p style='font-size: 1em; line-height: 1.5;'>{summary_response.content}</p>
    #         </div>
    #         """, unsafe_allow_html=True)
    # except Exception as e:
    #     st.error(f"Error generating summary: {str(e)}")

    # Create 5 columns for prediction results with adjusted ratios
    pred_col1, pred_col2, pred_col3, pred_col4, pred_col5 = st.columns(
        [2, 1.5, 1.5, 1.5, 1.5])

    with pred_col1:
        st.image(st.session_state.cropped_img_path,
                 caption="Analyzed Area", width=450, use_container_width=True)

    with pred_col2:
        # confidence = predictions['confidences']['rooftop_type']  # COMMENTED OUT: Percentage display
        st.markdown(f'''<div class="pred-card"><h4>🏠 Rooftop Analysis</h4>
        <p><b>Predicted:</b> <span class="predicted-value">{predictions['predictions']['rooftop_type'].title()}</span></p></div>''', unsafe_allow_html=True)

    with pred_col3:
        st.markdown(f'''
        <div class="pred-card"><h4>🏠 House Characteristics</h4><ul>
        <li><b>Floor Type:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['Floor'].title()}</span></li>
        <li><b>Wall Type:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['Wall'].title()}</span></li>
        <li><b>Water Supply:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['S.D.Water'].title()}</span></li>
        <li><b>Govt. House Scheme:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['Govt.H.Sch'].title()}</span></li>
        </ul></div>
        ''', unsafe_allow_html=True)

    with pred_col4:
        st.markdown(f'''
        <div class="pred-card"><h4>👥 Social Indicators</h4><ul>
        <li><b>Occupation:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['Occupation'].title()}</span></li>
        <li><b>Ration Card:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['Rationcard'].title()}</span></li>
        <li><b>MGNREGA:</b><br>
            Predicted: <span class="predicted-value">{predictions['predictions']['MGNREGA'].title()}</span></li>
        </ul></div>
        ''', unsafe_allow_html=True)

    with pred_col5:
        st.markdown(f'''
        <div class="pred-card"><h4>🚗 Vehicle Ownership</h4><ul>
        <li><b>Bike:</b><br>
            Predicted: <span class="predicted-value">{int(predictions['predictions']['Bike'])}</span></li>
        <li><b>Cycle:</b><br>
            Predicted: <span class="predicted-value">{int(predictions['predictions']['cycle'])}</span></li>
        <li><b>Car:</b><br>
            Predicted: <span class="predicted-value">{int(predictions['predictions']['car'])}</span></li>
        </ul></div>
        ''', unsafe_allow_html=True)

    # Add Ground Truth Section
    st.markdown("<br>", unsafe_allow_html=True)
    section_header("📋 Ground Truth Data:", color="blue-70")

    # Get actual data
    actual_data = None
    if st.session_state.tiff_file:
        actual_data = get_actual_data(st.session_state.tiff_file, selected_id)

    if actual_data:
        # Create 3 columns for ground truth data
        truth_col1, truth_col2, truth_col3 = st.columns([1.5, 1.5, 1.5])

        with truth_col1:
            # Compare house details
            rooftop_match = str(actual_data.get('rooftop_type', '')).lower(
            ) == predictions['predictions']['rooftop_type'].lower()
            floor_match = str(actual_data.get('Floor', '')).lower(
            ) == predictions['predictions']['Floor'].lower()
            wall_match = str(actual_data.get('Wall', '')).lower(
            ) == predictions['predictions']['Wall'].lower()
            water_match = str(actual_data.get('S.D.Water', '')).lower(
            ) == predictions['predictions']['S.D.Water'].lower()
            scheme_match = str(actual_data.get('Govt.H.Sch', '')).lower(
            ) == predictions['predictions']['Govt.H.Sch'].lower()

            st.markdown(f'''
            <div class="pred-card"><h4>🏠 House Details</h4><ul>
            <li><b>Rooftop Type:</b> <span style="color: {'#4CAF50' if rooftop_match else '#ff5252'}"><b>{str(actual_data.get('rooftop_type', 'N/A')).title()}</b></span></li>
            <li><b>Floor Type:</b> <span style="color: {'#4CAF50' if floor_match else '#ff5252'}"><b>{str(actual_data.get('Floor', 'N/A')).title()}</b></span></li>
            <li><b>Wall Type:</b> <span style="color: {'#4CAF50' if wall_match else '#ff5252'}"><b>{str(actual_data.get('Wall', 'N/A')).title()}</b></span></li>
            <li><b>Water Supply:</b> <span style="color: {'#4CAF50' if water_match else '#ff5252'}"><b>{str(actual_data.get('S.D.Water', 'N/A')).title()}</b></span></li>
            <li><b>Govt. House Scheme:</b> <span style="color: {'#4CAF50' if scheme_match else '#ff5252'}"><b>{str(actual_data.get('Govt.H.Sch', 'N/A')).title()}</b></span></li>
            </ul></div>
            ''', unsafe_allow_html=True)

        with truth_col2:
            # Compare social status
            occupation_match = str(actual_data.get('Occupation', '')).lower(
            ) == predictions['predictions']['Occupation'].lower()
            ration_match = str(actual_data.get('Rationcard', '')).lower(
            ) == predictions['predictions']['Rationcard'].lower()
            mgnrega_match = str(actual_data.get('MGNREGA', '')).lower(
            ) == predictions['predictions']['MGNREGA'].lower()

            st.markdown(f'''
            <div class="pred-card"><h4>👥 Social Status</h4><ul>
            <li><b>Occupation:</b> <span style="color: {'#4CAF50' if occupation_match else '#ff5252'}"><b>{str(actual_data.get('Occupation', 'N/A')).title()}</b></span></li>
            <li><b>Ration Card:</b> <span style="color: {'#4CAF50' if ration_match else '#ff5252'}"><b>{str(actual_data.get('Rationcard', 'N/A')).title()}</b></span></li>
            <li><b>MGNREGA:</b> <span style="color: {'#4CAF50' if mgnrega_match else '#ff5252'}"><b>{str(actual_data.get('MGNREGA', 'N/A')).title()}</b></span></li>
            </ul></div>
            ''', unsafe_allow_html=True)

        with truth_col3:
            # Compare vehicle ownership
            bike_match = str(actual_data.get('Bike', '')) == str(
                int(predictions['predictions']['Bike']))
            cycle_match = str(actual_data.get('Cycle', '')) == str(
                int(predictions['predictions']['cycle']))
            car_match = str(actual_data.get('Car', '')) == str(
                int(predictions['predictions']['car']))

            st.markdown(f'''
            <div class="pred-card"><h4>🚗 Vehicle Ownership</h4><ul>
            <li><b>Bike:</b> <span style="color: {'#4CAF50' if bike_match else '#ff5252'}"><b>{str(actual_data.get('Bike', 'N/A'))}</b></span></li>
            <li><b>Cycle:</b> <span style="color: {'#4CAF50' if cycle_match else '#ff5252'}"><b>{str(actual_data.get('Cycle', 'N/A'))}</b></span></li>
            <li><b>Car:</b> <span style="color: {'#4CAF50' if car_match else '#ff5252'}"><b>{str(actual_data.get('Car', 'N/A'))}</b></span></li>
            </ul></div>
            ''', unsafe_allow_html=True)

        # Add concise comparison analysis
        # COMMENTED OUT: LLM Comparison Analysis
        # comparison_prompt = f"""Compare the predicted values with ground truth data and provide a concise analysis in point format, change new line with each new point:
        # 
        # Predicted Values:
        # - Rooftop Type: {predictions['predictions']['rooftop_type'].title()} (Confidence: {predictions['confidences']['rooftop_type']:.1%})
        # - Floor Type: {predictions['predictions']['Floor'].title()} (Confidence: {predictions['confidences']['Floor']:.1%})
        # - Wall Type: {predictions['predictions']['Wall'].title()} (Confidence: {predictions['confidences']['Wall']:.1%})
        # - Water Supply: {predictions['predictions']['S.D.Water'].title()} (Confidence: {predictions['confidences']['S.D.Water']:.1%})
        # - Govt. House Scheme: {predictions['predictions']['Govt.H.Sch'].title()} (Confidence: {predictions['confidences']['Govt.H.Sch']:.1%})
        # - Occupation: {predictions['predictions']['Occupation'].title()} (Confidence: {predictions['confidences']['Occupation']:.1%})
        # - Ration Card: {predictions['predictions']['Rationcard'].title()} (Confidence: {predictions['confidences']['Rationcard']:.1%})
        # - MGNREGA: {predictions['predictions']['MGNREGA'].title()} (Confidence: {predictions['confidences']['MGNREGA']:.1%})
        # - Vehicles: {int(predictions['predictions']['Bike'])} bikes, {int(predictions['predictions']['cycle'])} cycles, {int(predictions['predictions']['car'])} cars
        # 
        # Ground Truth Values:
        # - Rooftop Type: {actual_data.get('rooftop_type', 'N/A').title()}
        # - Floor Type: {actual_data.get('Floor', 'N/A').title()}
        # - Wall Type: {actual_data.get('Wall', 'N/A').title()}
        # - Water Supply: {actual_data.get('S.D.Water', 'N/A').title()}
        # - Govt. House Scheme: {actual_data.get('Govt.H.Sch', 'N/A').title()}
        # - Occupation: {actual_data.get('Occupation', 'N/A').title()}
        # - Ration Card: {actual_data.get('Rationcard', 'N/A').title()}
        # - MGNREGA: {actual_data.get('MGNREGA', 'N/A').title()}
        # - Vehicles: {actual_data.get('Bike', 'N/A')} bikes, {actual_data.get('Cycle', 'N/A')} cycles, {actual_data.get('Car', 'N/A')} cars
        # 
        # Provide a concise 5 point analysis focusing on:
        # 1. Overall accuracy of predictions
        # 2. Key correct predictions
        # 3. Major discrepancies
        # 4. Model's strengths
        # 5. Areas for improvement"""

        # try:
        #     with st.spinner("Analyzing predictions..."):
        #         comparison_response = llm.invoke(comparison_prompt)
        #         # Split the response into lines and format as a list
        #         analysis_points = comparison_response.content.strip().split('\n')
        #         list_items_html = ''.join(
        #             [f'<li>{point.strip()[3:].strip() if point.strip() and point.strip()[0].isdigit() and point.strip()[1] == "." else point.strip()}</li>' for point in analysis_points if point.strip()])
        # 
        #         st.markdown(f"""
        #         <div style='margin: 20px 0; padding: 15px; background: #232323; border-radius: 8px;'>
        #             <h4 style='margin-top: 0; color: #4CAF50;'>📊 Model Performance Analysis</h4>
        #             <ul style='font-size: 1em; line-height: 1.5; padding-left: 20px;'>
        #                 {list_items_html}
        #             </ul>
        #         </div>
        #         """, unsafe_allow_html=True)
        # except Exception as e:
        #     st.error(f"Error generating comparison analysis: {str(e)}")
    else:
        st.warning(f"Ground truth data not found for ID: {selected_id}")

# Only show the download button if prediction is complete
show_download_button = False
if 'selected_mask' in st.session_state and st.session_state.selected_mask is not None and st.session_state.step == 3 and 'predictions' in st.session_state:
    show_download_button = True

# Show action buttons (Reset and Download) only after Excel is loaded
show_action_buttons_container = False
# Check if essential files are loaded before showing action buttons
if st.session_state.tiff_file is not None and st.session_state.excel_file is not None:
    show_action_buttons_container = True
