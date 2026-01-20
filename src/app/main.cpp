#include <meme/meme.hpp>

#include <igl/Timer.h>
#include <igl/read_triangle_mesh.h>
#include <CLI/CLI.hpp>
#include <filesystem>

#include <functional>
#include <iostream>
#include <optional>
#include <string>

#include <mshio/mshio.h>

class MshData
{
public:
    inline size_t get_num_face_vertices() const { return get_num_vertices(2); }

    inline size_t get_num_faces() const { return get_num_simplex_elements(2); }

    template <typename Fn>
    void extract_face_vertices(Fn&& set_vertex_cb) const
    {
        return extract_vertices(2, std::forward<Fn>(set_vertex_cb));
    }

    template <typename Fn>
    void extract_faces(Fn&& set_face_cb) const
    {
        extract_simplex_elements<2>(std::forward<Fn>(set_face_cb));
    }

    void load(const std::string& filename) { m_spec = mshio::load_msh(filename); }

    void load(std::istream& in) { m_spec = mshio::load_msh(in); }

private:
    const mshio::NodeBlock* get_vertex_block(const int dim) const
    {
        for (const auto& block : m_spec.nodes.entity_blocks) {
            if (block.entity_dim == dim) {
                return &block;
            }
        }
        return nullptr;
    }

    const mshio::ElementBlock* get_simplex_element_block(const int dim) const
    {
        for (const auto& block : m_spec.elements.entity_blocks) {
            if (block.entity_dim == dim) {
                return &block;
            }
        }
        return nullptr;
    }

    size_t get_num_vertices(const int dim) const
    {
        const auto* block = get_vertex_block(dim);
        if (block != nullptr) {
            return block->num_nodes_in_block;
        } else {
            return 0;
        }
    }

    size_t get_num_simplex_elements(const int dim) const
    {
        const auto* block = get_simplex_element_block(dim);
        if (block != nullptr) {
            return block->num_elements_in_block;
        } else {
            return 0;
        }
    }

    template <typename Fn>
    void extract_vertices(const int dim, Fn&& set_vertex_cb) const
    {
        const auto* block = get_vertex_block(dim);
        if (block == nullptr) return;

        const size_t num_vertices = block->num_nodes_in_block;
        if (num_vertices == 0) return;

        const size_t tag_offset = block->tags.front();
        for (size_t i = 0; i < num_vertices; i++) {
            size_t tag = block->tags[i] - tag_offset;
            set_vertex_cb(tag, block->data[i * 3], block->data[i * 3 + 1], block->data[i * 3 + 2]);
        }
    }

    template <int DIM, typename Fn>
    void extract_simplex_elements(Fn&& set_element_cb) const
    {
        const auto* vertex_block = get_vertex_block(DIM);
        const auto* element_block = get_simplex_element_block(DIM);
        if (element_block == nullptr) return;
        assert(vertex_block != nullptr);

        const size_t num_elements = element_block->num_elements_in_block;
        if (num_elements == 0) return;
        assert(vertex_block->num_nodes_in_block != 0);

        const size_t vert_tag_offset = vertex_block->tags.front();
        const size_t elem_tag_offset = element_block->data.front();
        for (size_t i = 0; i < num_elements; i++) {
            const size_t tag = element_block->data[i * (DIM + 2)] - elem_tag_offset;
            assert(tag < num_elements);
            const auto* element = element_block->data.data() + i * (DIM + 2) + 1;

            if constexpr (DIM == 1) {
                set_element_cb(tag, element[0] - vert_tag_offset, element[1] - vert_tag_offset);
            } else if constexpr (DIM == 2) {
                set_element_cb(
                    tag,
                    element[0] - vert_tag_offset,
                    element[1] - vert_tag_offset,
                    element[2] - vert_tag_offset);
            } else if constexpr (DIM == 3) {
                set_element_cb(
                    tag,
                    element[0] - vert_tag_offset,
                    element[1] - vert_tag_offset,
                    element[2] - vert_tag_offset,
                    element[3] - vert_tag_offset);
            }
        }
    }

public:
    mshio::MshSpec m_spec;
};


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

    if (mesh_file.extension() == ".msh") {
        MshData msh;
        msh.load(mesh_file.string());
        V.resize(msh.get_num_face_vertices(), 3);
        F.resize(msh.get_num_faces(), 3);
        msh.extract_face_vertices(
            [&V](size_t i, double x, double y, double z) { V.row(i) << x, y, z; });
        msh.extract_faces(
            [&F](size_t i, size_t v0, size_t v1, size_t v2) { F.row(i) << v0, v1, v2; });
    } else {
        igl::read_triangle_mesh(mesh_file.string(), V, F);
    }

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