import plotly.express as px
import pandas as pd


def timeline_figure(df: pd.DataFrame, title: str = "Agenda"):
    # df: doctor, patient, start_min, end_min, tipo('Normal'/'Urgencia')
    fig = px.timeline(
        df,
        x_start="start_min",
        x_end="end_min",
        y="doctor",
        color="tipo",
        hover_data=["patient", "slot", "urgent"],
        title=title,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(xaxis_title="Minutos desde inicio de jornada")
    return fig