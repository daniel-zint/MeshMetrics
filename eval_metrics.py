import argparse
import os
import plotly.graph_objs
import plotly.subplots

"""
This script reads a CSV file containing evaluation metrics for 3D meshes
and generates histogram plots for each metric, saving them to a specified output folder.

Usage:
    python eval_metrics.py -i path/to/metrics.csv -o path/to/output/folder

Arguments:
    -i: Path to the CSV file containing metrics (default: "metrics.csv").
    -o: Output folder for all plots (default: "plots").

Installation:
    pip install plotly
    pip install --upgrade kaleido

Performance Tips:
    - Use --format=html for fastest generation (no external rendering needed)
    - Use --format=png for raster images (slower, requires kaleido)
    - Parallel processing is automatic for multiple plots
    - Use --workers N to control parallelization (default: number of CPU cores)
"""


def get_metrics_from_csv(csv_path):
    """Reads a CSV file and returns the metrics as a list of strings."""
    with open(csv_path, "r") as f:
        lines = f.readlines()

    # Assuming the first line is the header and the other lines contain the metrics
    header = lines[0].strip().split(",")
    values = []
    for line in lines[1:]:
        values.append(line.strip().split(","))

    # Transpose values to get columns
    columns = list(zip(*values))

    metrics = {}
    for i, col in enumerate(columns):
        metrics[header[i]] = list(col)

    # rename headers
    ## in_max_min_angle,in_avg_min_angle,in_min_max_angle,in_max_max_angle,in_avg_max_angle,in_min_ratio,in_max_ratio,in_avg_ratio,in_min_shape,in_max_shape,in_avg_shape,in_min_edge,in_max_edge,in_avg_edge,in_#F,in_#V,in_has_zero_area,in_has_zero_edge,in_runtime_seconds,out_min_min_angle,out_max_min_angle,out_avg_min_angle,out_min_max_angle,out_max_max_angle,out_avg_max_angle,out_min_ratio,out_max_ratio,out_avg_ratio,out_min_shape,out_max_shape,out_avg_shape,out_min_edge,out_max_edge,out_avg_edge,out_#F,out_#V,out_has_zero_area,out_has_zero_edge,out_hausdorff_distance,out_runtime_seconds,mesh_file
    rename_dict = {
        "in_min_min_angle": "",
        "in_max_min_angle": "",
        "in_avg_min_angle": "Angle (before remeshing)",
        "in_min_max_angle": "",
        "in_max_max_angle": "",
        "in_avg_max_angle": "",
        "in_min_ratio": "",
        "in_max_ratio": "",
        "in_avg_ratio": "Radius ratio (before remeshing)",
        "in_min_shape": "",
        "in_max_shape": "",
        "in_avg_shape": "Shape regularity (before remeshing)",
        "in_min_edge": "",
        "in_max_edge": "",
        "in_avg_edge": "",
        "in_#F": "Faces (before remeshing)",
        "in_#V": "Vertices (before remeshing)",
        "in_has_zero_area": "",
        "in_has_zero_edge": "",
        "in_runtime_seconds": "",
        "out_min_min_angle": "",
        "out_max_min_angle": "",
        "out_avg_min_angle": "Angle (after remeshing)",
        "out_min_max_angle": "",
        "out_max_max_angle": "",
        "out_avg_max_angle": "",
        "out_min_ratio": "",
        "out_max_ratio": "",
        "out_avg_ratio": "Radius ratio (after remeshing)",
        "out_min_shape": "",
        "out_max_shape": "",
        "out_avg_shape": "Shape regularity (after remeshing)",
        "out_min_edge": "",
        "out_max_edge": "",
        "out_avg_edge": "",
        "out_#F": "Faces (after remeshing)",
        "out_#V": "Vertices (after remeshing)",
        "out_has_zero_area": "",
        "out_has_zero_edge": "",
        "out_hausdorff_distance": "Hausdorff distance",
        "out_runtime_seconds": "Runtime remeshing (seconds)",
        "mesh_file": "",
    }
    for old_name, new_name in rename_dict.items():
        if old_name in metrics:
            if new_name == "":
                metrics.pop(old_name)
            else:
                metrics[new_name] = metrics.pop(old_name)

    return metrics


def create_metric_plot(metric_name, values, color):
    """Create a single metric histogram plot."""
    fig = plotly.graph_objs.Figure()
    fig.add_trace(
        plotly.graph_objs.Histogram(
            x=list(map(float, values)), nbinsx=50, marker_color=color
        ),
    )

    fig.update_layout(
        xaxis_title=metric_name,
        margin=dict(l=40, r=40, t=40, b=40),
        font=dict(family="Times New Roman, serif", size=14),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(
            showgrid=True,
            gridcolor="lightgray",
            gridwidth=1,
        ),
        xaxis=dict(
            showgrid=False,
        ),
    )
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate and plot metrics from CSV.")
    parser.add_argument(
        "-i",
        type=str,
        default="metrics.csv",
        help="Path to the CSV file containing metrics.",
    )
    parser.add_argument(
        "-o",
        type=str,
        default="plots",
        help="Output folder for all plots.",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="png",
        choices=["html", "png", "svg"],
        help="Output format: html (fastest), png/svg (slower, requires kaleido). Default: png",
    )

    args = parser.parse_args()

    csv_path = args.i
    output_folder = args.o
    image_format = args.format

    metrics = get_metrics_from_csv(csv_path)

    # Set3 colormap colors
    set3_colors = [
        "#8DD3C7",
        # "#FFFFB3",
        "#BEBADA",
        "#FB8072",
        "#80B1D3",
        "#FDB462",
        "#B3DE69",
        "#FCCDE5",
        # "#D9D9D9",
        "#BC80BD",
        "#CCEBC5",
        "#FFED6F",
    ]

    # create a plot for each metric of the csv and store it in output folder
    print(
        f"Generating plots for metrics in {output_folder} (format: {image_format})..."
    )
    os.makedirs(output_folder, exist_ok=True)

    for i, (metric_name, values) in enumerate(metrics.items()):
        color = set3_colors[i % len(set3_colors)]
        fig = create_metric_plot(metric_name, values, color)

        if image_format == "html":
            plot_file = os.path.join(output_folder, f"{metric_name}.html")
            fig.write_html(plot_file)
        elif image_format == "png":
            plot_file = os.path.join(output_folder, f"{metric_name}.png")
            fig.write_image(plot_file, width=1000, height=600)
        elif image_format == "svg":
            plot_file = os.path.join(output_folder, f"{metric_name}.svg")
            fig.write_image(plot_file, width=1000, height=600)

        print(f"{i}: Saved {plot_file}")

    print(f"Plots saved to {output_folder}")
