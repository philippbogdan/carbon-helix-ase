"""Patch the vendored Frontier native engine (adds update_priors) and compile it as
hase._hase_frontier_native (see third_party/frontier/README.md)."""
import os
import sys
import sysconfig
import subprocess

root = os.path.join(os.path.dirname(__file__), "..")
src = open(os.path.join(root, "third_party", "frontier", "_frontier_native.upstream.cpp")).read()

FUNC = r'''
void refresh_local_pattern_table(Column& column) {
    LocalPatternTable& table = column.pattern_table;
    if (!table.enabled) {
        return;
    }
    const int nr = table.row_count;
    double bw[12] = {0.0};
    double aw[12] = {0.0};
    auto find_row = [&](const RowTerm& term) -> int {
        for (int r = 0; r < nr; ++r) {
            if (table.rows[static_cast<std::size_t>(r)].limb == term.limb &&
                table.rows[static_cast<std::size_t>(r)].bit == term.bit) {
                return r;
            }
        }
        throw std::runtime_error("refresh_local_pattern_table: row not found");
    };
    for (const RowTerm& term : column.before_terms) bw[find_row(term)] += term.parity;
    for (const RowTerm& term : column.after_terms) aw[find_row(term)] += term.parity;
    const std::size_t size = static_cast<std::size_t>(1ULL << static_cast<std::size_t>(nr));
    double total_after = 0.0;
    for (int r = 0; r < nr; ++r) total_after += aw[r];
    std::vector<double> nod(size), tgd(size);
    nod[0] = 0.0;
    tgd[0] = total_after;
    for (std::size_t pattern = 1; pattern < size; ++pattern) {
        const int i = __builtin_ctzll(static_cast<unsigned long long>(pattern));
        const std::size_t q = pattern & (pattern - 1);
        nod[pattern] = nod[q] + (aw[i] - bw[i]);
        tgd[pattern] = tgd[q] - aw[i] - bw[i];
    }
    for (std::size_t pattern = 0; pattern < size; ++pattern) {
        if (table.small_enabled) {
            table.no_delta_small[pattern] = nod[pattern];
            table.toggle_delta_small[pattern] = tgd[pattern];
            table.no_delta_small_s1024[pattern] = quantize_metric(nod[pattern], 1024);
            table.toggle_delta_small_s1024[pattern] = quantize_metric(tgd[pattern], 1024);
        } else {
            table.no_delta[pattern] = nod[pattern];
            table.toggle_delta[pattern] = tgd[pattern];
            table.no_delta_s1024[pattern] = quantize_metric(nod[pattern], 1024);
            table.toggle_delta_s1024[pattern] = quantize_metric(tgd[pattern], 1024);
        }
    }
}

PyObject* py_update_priors(PyObject*, PyObject* args) {
    PyObject* capsule = nullptr;
    PyObject* objs[4] = {nullptr, nullptr, nullptr, nullptr};
    if (!PyArg_ParseTuple(args, "OOOOO", &capsule, &objs[0], &objs[1], &objs[2], &objs[3])) {
        return nullptr;
    }
    Py_buffer bufs[4];
    int got = 0;
    try {
        NativeModel* model = model_from_capsule(capsule);
        for (; got < 4; ++got) {
            if (PyObject_GetBuffer(objs[got], &bufs[got], PyBUF_C_CONTIGUOUS) < 0) {
                throw std::runtime_error("update_priors: arguments must be contiguous float64 buffers");
            }
            if (bufs[got].itemsize != 8 && bufs[got].len != 0) {
                throw std::runtime_error("update_priors: buffers must hold float64 values");
            }
        }
        const double* no_err = static_cast<const double*>(bufs[0].buf);
        const double* tog = static_cast<const double*>(bufs[1].buf);
        const double* bef = static_cast<const double*>(bufs[2].buf);
        const double* aft = static_cast<const double*>(bufs[3].buf);
        const std::size_t ncol = model->columns.size();
        if (static_cast<std::size_t>(bufs[0].len / 8) != ncol || static_cast<std::size_t>(bufs[1].len / 8) != ncol) {
            throw std::runtime_error("update_priors: wrong number of columns");
        }
        std::size_t ib = 0, ia = 0;
        const std::size_t nb = static_cast<std::size_t>(bufs[2].len / 8);
        const std::size_t na = static_cast<std::size_t>(bufs[3].len / 8);
        for (std::size_t c = 0; c < ncol; ++c) {
            Column& column = model->columns[c];
            if (ib + column.before_terms.size() > nb || ia + column.after_terms.size() > na) {
                throw std::runtime_error("update_priors: parity buffers too short");
            }
            bool same = (column.no_error_log_const == no_err[c]) && (column.toggle_logodds == tog[c]);
            for (std::size_t k = 0; same && k < column.before_terms.size(); ++k) {
                same = column.before_terms[k].parity == bef[ib + k];
            }
            for (std::size_t k = 0; same && k < column.after_terms.size(); ++k) {
                same = column.after_terms[k].parity == aft[ia + k];
            }
            if (same) {
                ib += column.before_terms.size();
                ia += column.after_terms.size();
                continue;
            }
            column.no_error_log_const = no_err[c];
            column.toggle_logodds = tog[c];
            column.no_error_log_const_s1024 = quantize_metric(column.no_error_log_const, 1024);
            column.toggle_logodds_s1024 = quantize_metric(column.toggle_logodds, 1024);
            for (RowTerm& term : column.before_terms) {
                term.parity = bef[ib++];
                term.parity_s1024 = quantize_metric(term.parity, 1024);
            }
            for (RowTerm& term : column.after_terms) {
                term.parity = aft[ia++];
                term.parity_s1024 = quantize_metric(term.parity, 1024);
            }
            refresh_local_pattern_table(column);
        }
        if (ib != nb || ia != na) {
            throw std::runtime_error("update_priors: parity buffers have the wrong length");
        }
        for (int k = 0; k < got; ++k) PyBuffer_Release(&bufs[k]);
        Py_RETURN_NONE;
    } catch (const std::exception& exc) {
        for (int k = 0; k < got; ++k) PyBuffer_Release(&bufs[k]);
        PyErr_SetString(PyExc_ValueError, exc.what());
        return nullptr;
    }
}

PyMethodDef methods[] = {
    {"update_priors", py_update_priors, METH_VARARGS, "Replace column priors and pruning weights in place."},'''



def patched_source():
    s2 = src
    assert "PyMethodDef methods[] = {" in s2
    s2 = s2.replace("PyMethodDef methods[] = {", FUNC, 1)
    s2 = s2.replace('"_frontier_native",', '"_hase_frontier_native",')
    s2 = s2.replace("PyInit__frontier_native", "PyInit__hase_frontier_native")
    return s2


def build(out_dir, build_dir=None):
    build_dir = build_dir or out_dir
    os.makedirs(build_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    cpp = os.path.join(build_dir, "_hase_frontier_native.cpp")
    with open(cpp, "w") as fh:
        fh.write(patched_source())
    ext = sysconfig.get_config_var("EXT_SUFFIX")
    out = os.path.join(out_dir, "_hase_frontier_native" + ext)
    inc = sysconfig.get_paths()["include"]
    cxx = os.environ.get("CXX", "c++")
    flags = ["-O3", "-std=c++17", "-shared", "-fPIC", "-DNDEBUG", f"-I{inc}"]
    if sys.platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    tmp = out + ".tmp%d" % os.getpid()
    cmd = [cxx] + flags + [cpp, "-o", tmp]
    subprocess.check_call(cmd)
    os.replace(tmp, out)
    return out


if __name__ == "__main__":
    out = build(os.path.join(root, "src", "hase"), os.path.join(root, ".tmp", "native_build"))
    print("built", out)
