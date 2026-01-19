#include <meme/meme.hpp>

#include <igl/Timer.h>
#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
#include <filesystem>

int main(int argc, char** argv)
{
    CLI::App app{argv[0]};
    std::filesystem::path mesh_file;

    app.add_option("-i", mesh_file)->required()->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    meme::MatrixXd V;
    meme::MatrixXi F;

    igl::Timer timer;
    timer.start();

    igl::read_triangle_mesh(mesh_file.string(), V, F);

    // simple statistics
    {
        const auto metrics = meme::get_metrics(V, F);
        const auto metrics_names = meme::get_metrics_names();

        static_assert(metrics.size() == metrics_names.size());

        for (size_t i = 0; i < metrics.size(); ++i) {
            std::cout << metrics_names[i] << ": " << metrics[i] << std::endl;
        }
    }

    // per triangle
    {
        const auto metrics = meme::get_metrics_per_tri(V, F);
        const auto metrics_names = meme::get_metrics_names_per_tri();

        for (size_t i = 0; i < metrics_names.size(); ++i) {
            std::cout << metrics_names[i] << ", ";
        }
        std::cout << std::endl;
        if (metrics.rows() > 20) {
            std::cout << metrics.block(0, 0, 20, metrics.cols()) << std::endl;
            for (size_t i = 0; i < metrics_names.size(); ++i) {
                std::cout << "...\t";
            }
            std::cout << std::endl;
        } else {
            std::cout << metrics << std::endl;
        }
    }
    // edge lengths
    {
        const auto ls = meme::get_relative_edge_lengths(V, F);
        std::cout << ls.rows() << " edges: \n\t";
        if (ls.rows() > 20) {
            std::cout << ls.block(0, 0, 20, 1).transpose();
        } else {
            std::cout << ls.transpose();
        }
        std::cout << std::endl;
    }

    timer.stop();

    std::cout << "Took " << timer.getElapsedTimeInSec() << " seconds" << std::endl;
}