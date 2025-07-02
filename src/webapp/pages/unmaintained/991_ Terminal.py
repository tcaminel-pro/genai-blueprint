from streamlit_ttyd import terminal

# st.text("Terminal showing processes running on your system using the top command")

# # start the ttyd server and display the terminal on streamlit
ttydprocess, port = terminal(cmd="nano", readonly=True)


# time.sleep(60)
# ttydprocess.kill()
ttydprocess1, port1 = terminal()
# time.sleep(60)
# ttydprocess1.kill()
