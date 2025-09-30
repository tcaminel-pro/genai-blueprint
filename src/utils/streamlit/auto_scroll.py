"""
Add autoscrolling to a Streamlit container

"""
# taken from
# -  https://discuss.streamlit.io/t/need-help-with-the-automatic-scroll-down-to-see-the-latest-result-of-conversation/91657/4
# -  https://stackoverflow.com/a/79246126
# Tried several other methods, this is the only one working (so far)

import random

from streamlit.components.v1 import html
from streamlit.delta_generator import DeltaGenerator


def scroll_to_here() -> DeltaGenerator:
    """
    Add autoscrolling to a container.

    Usage :
    ```
    content_container = st.container(height=200)
    if "messages" not in st.session_state:
        st.session_state.messages = []
        st.session_state.counter = 0
    if st.button("Add New Message"):
        st.session_state.counter += 1
        st.session_state.messages.append(f"Message #{st.session_state.counter}")
    with content_container:
        for msg in st.session_state.messages:
            st.text(msg)
        scroll_to_here()
    ```
    or:
    ```
    if st.button("Scroll to bottom"):
        content()
        scroll_to_here()
    else:
        content()
    """
    return html(
        f"""
        <div id="scroll-to-here" style='background: cyan; height=1px;' />
        <script id="{random.randint(1000, 9999)}">
            var div = document.getElementById('scroll-to-here');
            if (div) {{
                div.scrollIntoView({{ behavior: 'smooth', block: 'end' }});
                div.remove();
            }}
        </script>
        """,
        height=1,
    )
