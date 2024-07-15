#  Attempts to remove CrewAI Telemety
# See https://github.com/crewAIInc/crewAI/issues/372


from crewai.telemetry import Telemetry


def noop(*args, **kwargs):
    pass


def disable_crewai_telemetry():
    for attr in dir(Telemetry):
        if callable(getattr(Telemetry, attr)) and not attr.startswith("__"):
            setattr(Telemetry, attr, noop)
