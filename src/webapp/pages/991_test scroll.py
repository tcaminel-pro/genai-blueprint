import time
import uuid

import streamlit as st

# Set page title
st.title("Auto-Scrolling Container Demo")

# Create a unique key for the session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.debug_info = []
    st.session_state.counter = 0

# Add debug information section
debug_expander = st.expander("Debug Information")
debug_container = debug_expander.container()

# Function to add a new message
def add_message():
    timestamp = time.strftime("%H:%M:%S")
    message_id = str(uuid.uuid4())[:8]
    st.session_state.counter += 1
    new_message = f"Message #{st.session_state.counter} - {timestamp} (ID: {message_id})"
    st.session_state.messages.append(new_message)

    # Add debug info
    debug_info = f"Added message: {new_message}"
    st.session_state.debug_info.append(debug_info)

    # Keep only the last 20 debug messages
    if len(st.session_state.debug_info) > 20:
        st.session_state.debug_info = st.session_state.debug_info[-20:]

# Button to add content
if st.button("Add New Message"):
    add_message()

# Create a container with fixed height for scrolling
st.markdown("""
<style>
    .scrollable-container {
        height: 200px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        margin-bottom: 10px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)

# Create a unique div ID for our scrollable container
container_id = "scrollable-messages-container"

# Start of HTML container
st.markdown(f'<div id="{container_id}" class="scrollable-container">', unsafe_allow_html=True)

# Add all messages as HTML
for msg in st.session_state.messages:
    st.markdown(f'<div class="message">{msg}</div>', unsafe_allow_html=True)

# End of HTML container
st.markdown('</div>', unsafe_allow_html=True)

# Auto-scroll JavaScript
if st.session_state.messages:
    st.markdown(
        f"""
        <script>
            // Function to scroll to the bottom of our specific container
            function scrollToBottom() {{
                const container = document.getElementById('{container_id}');
                if (container) {{
                    container.scrollTop = container.scrollHeight;
                    console.log('Scrolled container to bottom, height:', container.scrollHeight);
                }} else {{
                    console.log('Container not found: {container_id}');
                }}
            }}
            
            // Execute scroll immediately and after a delay to ensure content is loaded
            scrollToBottom();
            setTimeout(scrollToBottom, 100);
            setTimeout(scrollToBottom, 300);
        </script>
        """,
        unsafe_allow_html=True,
    )

# Display debug information
with debug_container:
    st.write(f"Total messages: {len(st.session_state.messages)}")
    st.write(f"Last action timestamp: {time.strftime('%H:%M:%S')}")
    st.write(f"Container ID: {container_id}")

    if st.session_state.debug_info:
        st.write("Recent actions:")
        for info in st.session_state.debug_info:
            st.text(info)
