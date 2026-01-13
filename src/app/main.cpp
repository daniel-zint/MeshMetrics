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

    const auto metrics = meme::get_metrics(V, F);
    const auto metrics_names = meme::get_metrics_names();

    static_assert(metrics.size() == metrics_names.size());

    for (size_t i = 0; i < metrics.size(); ++i) {
        std::cout << metrics_names[i] << ": " << metrics[i] << std::endl;
    }

    timer.stop();

    std::cout << "Took " << timer.getElapsedTimeInSec() << " seconds" << std::endl;
}