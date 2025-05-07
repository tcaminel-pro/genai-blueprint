import time
import uuid

import streamlit as st

# Set page title
st.title("Auto-Scrolling Container Demo")

# Create a container to hold the content
content_container = st.container(height=200)

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

# Display all messages in the container
with content_container:
    for msg in st.session_state.messages:
        st.text(msg)

    # Auto-scroll to bottom using JavaScript
    if st.session_state.messages:
        st.markdown(
            """
            <script>
                // Function to scroll to the bottom of a container
                function scrollToBottom() {
                    const containers = document.getElementsByClassName('stText');
                    if (containers.length > 0) {
                        const lastContainer = containers[containers.length - 1];
                        lastContainer.scrollIntoView({ behavior: 'smooth' });
                        
                        // Debug info for console
                        console.log('Auto-scrolled to:', lastContainer);
                    } else {
                        console.log('No containers found to scroll to');
                    }
                }
                
                // Execute scroll after a short delay to ensure DOM is updated
                setTimeout(scrollToBottom, 100);
            </script>
            """,
            unsafe_allow_html=True,
        )

# Display debug information
with debug_container:
    st.write(f"Total messages: {len(st.session_state.messages)}")
    st.write(f"Last action timestamp: {time.strftime('%H:%M:%S')}")

    if st.session_state.debug_info:
        st.write("Recent actions:")
        for info in st.session_state.debug_info:
            st.text(info)
