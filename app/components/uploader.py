import streamlit as st
import os

from src.security.file_validation import validate_pdf_envelope

MAX_FILE_SIZE_MB = 50

def render_upload_zone():
    """
    Renders the secure file upload zone for PDF files.
    Validates MIME type and file size.
    Returns:
        UploadedFile object if a valid PDF is uploaded, None otherwise.
    """
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### 📥 Upload PDF for Analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop or browse for a PDF file", 
        type=['pdf'], 
        accept_multiple_files=False,
        help=f"Maximum file size: {MAX_FILE_SIZE_MB}MB"
    )
    
    if uploaded_file is not None:
        # File size check. The inference layer repeats this check and returns an
        # explicit uncertain/abstain outcome if the envelope changes.
        file_size_mb = uploaded_file.size / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.error(f"❌ File size ({file_size_mb:.1f}MB) exceeds the maximum limit of {MAX_FILE_SIZE_MB}MB.")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
            
        # Server-side MIME type validation
        try:
            validation = validate_pdf_envelope(uploaded_file.getvalue())
            uploaded_file.seek(0)
            if not validation.valid:
                st.warning(
                    f"⚠️ PDF envelope issue ({validation.reason}); analysis will "
                    "fail closed with an uncertain/abstain outcome."
                )
                
            # Sanitize filename for display/logging
            safe_filename = os.path.basename(uploaded_file.name)
            st.success(f"✅ Successfully loaded: `{safe_filename}` ({file_size_mb:.2f} MB)")
            
            st.markdown('</div>', unsafe_allow_html=True)
            return uploaded_file
            
        except Exception as e:
            st.error(f"❌ Error validating file: {str(e)}")
            st.markdown('</div>', unsafe_allow_html=True)
            return None
            
    st.markdown('</div>', unsafe_allow_html=True)
    return None
