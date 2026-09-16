#include "similarity_io.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <stdexcept>
#include <system_error>

#if defined(_WIN32)
#  include <Windows.h>
#  include <fcntl.h>
#  include <io.h>
#  include <sys/stat.h>
#else
#  include <fcntl.h>
#  include <sys/stat.h>
#  include <sys/types.h>
#  include <unistd.h>
#endif

namespace parallel_rdkit {
namespace {

using size_type = std::size_t;

size_type checked_add(size_type a, size_type b, const char *what) {
    if (b > std::numeric_limits<size_type>::max() - a)
        throw std::overflow_error(what);
    return a + b;
}

size_type checked_mul(size_type a, size_type b, const char *what) {
    if (b && a > std::numeric_limits<size_type>::max() / b)
        throw std::overflow_error(what);
    return a * b;
}

std::filesystem::path parent_for(const std::filesystem::path& path) {
    auto parent = path.parent_path();
    return parent.empty() ? std::filesystem::path(".") : parent;
}

struct TemporaryFile {
    std::filesystem::path path;
    int fd = -1;
};

TemporaryFile make_temporary(const std::filesystem::path& parent,
                             const std::string& name) {
    // O_EXCL is important here: a collision must never make us open an
    // unrelated file.  The directory is the destination directory so that a
    // later rename/link is atomic.
    std::random_device device;
    std::mt19937_64 random(device());
    for (unsigned int attempt = 0; attempt != 100; ++attempt) {
        auto candidate = parent / ("." + name + "." +
                                   std::to_string(random()) + ".tmp");
#if defined(_WIN32)
        int fd = _open(candidate.string().c_str(),
                       _O_BINARY | _O_RDWR | _O_CREAT | _O_EXCL,
                       _S_IREAD | _S_IWRITE);
#else
        int fd = ::open(candidate.c_str(), O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC,
                        S_IRUSR | S_IWUSR);
#endif
        if (fd >= 0)
            return {std::move(candidate), fd};
        if (errno != EEXIST)
            throw std::system_error(errno, std::generic_category(),
                                    "create temporary similarity file");
    }
    throw std::runtime_error("could not create a unique temporary similarity file");
}

void close_fd(int fd) noexcept {
#if defined(_WIN32)
    if (fd >= 0) _close(fd);
#else
    if (fd >= 0) ::close(fd);
#endif
}

std::FILE *open_stream(int fd) {
#if defined(_WIN32)
    return _fdopen(fd, "w+b");
#else
    return ::fdopen(fd, "w+b");
#endif
}

int stream_fd(std::FILE *stream) {
#if defined(_WIN32)
    return _fileno(stream);
#else
    return fileno(stream);
#endif
}

void flush_durable(std::FILE *stream, int fd) {
    if (std::fflush(stream) != 0)
        throw std::system_error(errno, std::generic_category(), "flush similarity file");
#if defined(_WIN32)
    if (_commit(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "flush similarity file");
#else
    if (::fsync(fd) != 0)
        throw std::system_error(errno, std::generic_category(), "flush similarity file");
#endif
}

void write_all(std::FILE *stream, const void *data, size_type size) {
    const auto *bytes = static_cast<const unsigned char *>(data);
    while (size) {
        const size_type written = std::fwrite(bytes, 1, size, stream);
        if (!written) {
            if (std::ferror(stream))
                throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                        "write similarity file");
            throw std::runtime_error("short write to similarity file");
        }
        bytes += written;
        size -= written;
    }
}

std::string npy_header(size_type rows, size_type columns, size_type &data_offset) {
    std::string dictionary = "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(rows) + ", " + std::to_string(columns) + "), }";

    // The header is padded so the data begins on a 64-byte boundary, as
    // required by the NumPy format.  Select v1 whenever its uint16 header
    // length can represent the result; otherwise use v2's uint32 length.
    for (unsigned char version = 1; version <= 2; ++version) {
        const size_type prefix = version == 1 ? 10 : 12;
        const size_type base = checked_add(prefix, dictionary.size(),
                                            "NumPy header size overflow");
        const size_type with_newline = checked_add(base, 1,
                                                   "NumPy header size overflow");
        const size_type rounded = checked_add(with_newline, 63,
                                              "NumPy header size overflow") / 64 * 64;
        const size_type header_length = rounded - prefix;
        const size_type limit = version == 1 ?
            static_cast<size_type>(std::numeric_limits<std::uint16_t>::max()) :
            static_cast<size_type>(std::numeric_limits<std::uint32_t>::max());
        if (header_length > limit)
            continue;

        std::string header = dictionary;
        header.append(header_length - dictionary.size() - 1, ' ');
        header.push_back('\n');
        std::string result("\x93NUMPY", 6);
        result.push_back(static_cast<char>(version));
        result.push_back('\0');
        if (version == 1) {
            const auto length = static_cast<std::uint16_t>(header_length);
            result.push_back(static_cast<char>(length & 0xff));
            result.push_back(static_cast<char>((length >> 8) & 0xff));
        } else {
            result.push_back(static_cast<char>(header_length & 0xff));
            result.push_back(static_cast<char>((header_length >> 8) & 0xff));
            result.push_back(static_cast<char>((header_length >> 16) & 0xff));
            result.push_back(static_cast<char>((header_length >> 24) & 0xff));
        }
        result += header;
        data_offset = result.size();
        return result;
    }
    throw std::overflow_error("NumPy header is too large");
}

void seek_to(std::FILE *stream, size_type offset) {
#if defined(_WIN32)
    using offset_type = __int64;
    if (offset > static_cast<size_type>(std::numeric_limits<offset_type>::max()))
        throw std::overflow_error("similarity file offset overflow");
    if (_fseeki64(stream, static_cast<offset_type>(offset), SEEK_SET) != 0)
#else
    if (offset > static_cast<size_type>(std::numeric_limits<off_t>::max()))
        throw std::overflow_error("similarity file offset overflow");
    if (fseeko(stream, static_cast<off_t>(offset), SEEK_SET) != 0)
#endif
        throw std::system_error(errno, std::generic_category(),
                                "seek similarity file");
}

void publish(const std::filesystem::path& temporary,
             const std::filesystem::path& destination, bool overwrite) {
#if defined(_WIN32)
    if (overwrite) {
        if (!MoveFileExA(temporary.string().c_str(), destination.string().c_str(),
                         MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
            throw std::system_error(static_cast<int>(GetLastError()),
                                    std::system_category(),
                                    "publish similarity file");
    } else {
        // create_hard_link is an atomic no-clobber publication operation.
        std::error_code ec;
        std::filesystem::create_hard_link(temporary, destination, ec);
        if (ec) throw std::filesystem::filesystem_error("publish similarity file",
                                                         destination, ec);
        std::filesystem::remove(temporary, ec);
        if (ec) throw std::filesystem::filesystem_error("remove temporary similarity file",
                                                         temporary, ec);
    }
#else
    if (overwrite) {
        if (::rename(temporary.c_str(), destination.c_str()) != 0)
            throw std::system_error(errno, std::generic_category(),
                                    "publish similarity file");
    } else {
        std::error_code ec;
        std::filesystem::create_hard_link(temporary, destination, ec);
        if (ec) throw std::filesystem::filesystem_error("publish similarity file",
                                                         destination, ec);
        std::filesystem::remove(temporary, ec);
        if (ec) throw std::filesystem::filesystem_error("remove temporary similarity file",
                                                         temporary, ec);
    }
#endif
}

std::vector<std::string> subrange(const std::vector<std::string>& values,
                                  size_type start, size_type count) {
    return {values.begin() + static_cast<std::ptrdiff_t>(start),
            values.begin() + static_cast<std::ptrdiff_t>(start + count)};
}

} // namespace

std::tuple<std::vector<std::uint8_t>, std::vector<std::uint8_t>>
write_similarity_dense_npy(const std::vector<std::string>& left,
                          const std::vector<std::string>& right,
                          const FingerprintOptions& opts,
                          bool assume_sanitized,
                          size_type batch_size,
                          size_type tile_size,
                          const std::string& output_path,
                          bool overwrite) {
    if (!batch_size || !tile_size)
        throw std::invalid_argument("batch and tile sizes must be positive");
    const size_type pairs = checked_mul(left.size(), right.size(),
                                        "similarity matrix size overflow");
    const size_type payload = checked_mul(pairs, sizeof(float),
                                          "similarity file size overflow");

    // Validate options before creating any output. Libraries are prepared once
    // per batch below, retaining the left batch across all right batches.
    validate_similarity_options(opts);
    std::vector<std::uint8_t> left_valid(left.size(), 0);
    std::vector<std::uint8_t> right_valid(right.size(), 0);

    const std::filesystem::path destination(output_path);
    const auto parent = parent_for(destination);
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);
    if (ec) throw std::filesystem::filesystem_error("create similarity output directory",
                                                     parent, ec);
    auto temporary = make_temporary(parent, destination.filename().string());
    std::FILE *stream = nullptr;
    bool published = false;
    try {
        stream = open_stream(temporary.fd);
        if (!stream) {
            const int error = errno ? errno : EIO;
            close_fd(temporary.fd);
            temporary.fd = -1;
            throw std::system_error(error, std::generic_category(),
                                    "open temporary similarity file");
        }
        temporary.fd = -1; // owned by stream now

        size_type data_offset = 0;
        const auto header = npy_header(left.size(), right.size(), data_offset);
        write_all(stream, header.data(), header.size());
        const size_type file_size = checked_add(data_offset, payload,
                                                "similarity file size overflow");
#if defined(_WIN32)
        if (file_size > static_cast<size_type>(std::numeric_limits<__int64>::max()))
            throw std::overflow_error("similarity file size overflow");
        if (_chsize_s(stream_fd(stream), static_cast<__int64>(file_size)) != 0)
#else
        if (file_size > static_cast<size_type>(std::numeric_limits<off_t>::max()))
            throw std::overflow_error("similarity file size overflow");
        if (ftruncate(stream_fd(stream), static_cast<off_t>(file_size)) != 0)
#endif
            throw std::system_error(errno ? errno : EIO, std::generic_category(),
                                    "allocate similarity file");

        // Retain one prepared left batch while visiting each prepared right
        // batch. Only one tile buffer is live, so score scratch is bounded by
        // tile_size * tile_size independently of batch_size.
        if (left.empty()) {
            for (size_type j = 0; j < right.size();) {
                const size_type right_count = std::min(batch_size, right.size() - j);
                const auto right_batch = subrange(right, j, right_count);
                const auto right_library = prepare_similarity_library(right_batch, opts,
                                                                       assume_sanitized);
                std::copy(right_library.valid.begin(), right_library.valid.end(),
                          right_valid.begin() + static_cast<std::ptrdiff_t>(j));
                j += right_count;
            }
        }
        for (size_type i = 0; i < left.size();) {
            const size_type left_count = std::min(batch_size, left.size() - i);
            const auto left_batch = subrange(left, i, left_count);
            const auto left_library = prepare_similarity_library(left_batch, opts,
                                                                  assume_sanitized);
            std::copy(left_library.valid.begin(), left_library.valid.end(),
                      left_valid.begin() + static_cast<std::ptrdiff_t>(i));
            for (size_type j = 0; j < right.size();) {
                const size_type right_count = std::min(batch_size, right.size() - j);
                const auto right_batch = subrange(right, j, right_count);
                const auto right_library = prepare_similarity_library(right_batch, opts,
                                                                       assume_sanitized);
                std::copy(right_library.valid.begin(), right_library.valid.end(),
                          right_valid.begin() + static_cast<std::ptrdiff_t>(j));
                for (size_type x0 = 0; x0 < left_count; x0 += tile_size) {
                    const size_type tile_rows = std::min(tile_size, left_count - x0);
                    for (size_type y0 = 0; y0 < right_count; y0 += tile_size) {
                        const size_type tile_columns = std::min(tile_size, right_count - y0);
                        const size_type tile_size_elements = checked_mul(
                            tile_rows, tile_columns, "similarity tile size overflow");
                        std::vector<float> tile(tile_size_elements);
                        compute_similarity_packed_tile(left_library, x0,
                            right_library, y0, tile_rows, tile_columns,
                            tile_columns, tile.data());
                        for (size_type x = 0; x < tile_rows; ++x) {
                            const size_type element = checked_add(
                                checked_mul(i + x0 + x, right.size(),
                                            "similarity offset overflow"),
                                j + y0, "similarity offset overflow");
                            const size_type byte_offset = checked_add(
                                data_offset, checked_mul(element, sizeof(float),
                                                         "similarity offset overflow"),
                                "similarity offset overflow");
                            seek_to(stream, byte_offset);
                            write_all(stream, tile.data() + x * tile_columns,
                                      checked_mul(tile_columns, sizeof(float),
                                                  "similarity row size overflow"));
                        }
                    }
                }
                j += right_count;
            }
            i += left_count;
        }
        flush_durable(stream, stream_fd(stream));
        if (std::fclose(stream) != 0) {
            stream = nullptr;
            throw std::system_error(errno, std::generic_category(),
                                    "close similarity file");
        }
        stream = nullptr;
        publish(temporary.path, destination, overwrite);
        published = true;
    } catch (...) {
        if (stream) std::fclose(stream);
        if (temporary.fd >= 0) close_fd(temporary.fd);
        if (!published) {
            std::error_code remove_error;
            std::filesystem::remove(temporary.path, remove_error);
        }
        throw;
    }
    return {std::move(left_valid), std::move(right_valid)};
}

} // namespace parallel_rdkit
