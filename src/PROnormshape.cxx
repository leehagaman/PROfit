/**
 * @file PROnormshape.cxx
 * @brief Implementation of the norm-shape grid scan. See inc/PROnormshape.h.
 * @author PROfit Collaboration
 */
#include "PROnormshape.h"
#include "PROcess.h"
#include "PROtocall.h"
#include "PROlog.h"

#include <fstream>
#include <future>

namespace PROfit {

PROnormshape::PROnormshape(PROmetric &metric, const PROconfig &config, const PROpeller &prop,
                           size_t truth_var, const std::string &channel_name,
                           const std::vector<size_t> &innbins,
                           const std::vector<float> &los, const std::vector<float> &his,
                           const std::string &detector_name, float fixed_baseline)
    : metric(metric), config(config), prop(prop), truth_var(truth_var),
      channel_name(channel_name), detector_name(detector_name),
      fixed_baseline(fixed_baseline), nbins(innbins) {

    const PROmodel &model = metric.GetModel();
    const size_t nphys = model.nparams;

    if(nbins.size() != nphys || los.size() != nphys || his.size() != nphys) {
        log<LOG_ERROR>(L"%1% || Expected %2% entries (one per physics parameter) in nbins/lo/hi, got %3%/%4%/%5%. Terminating.")
            % __func__ % nphys % nbins.size() % los.size() % his.size();
        exit(EXIT_FAILURE);
    }

    for(size_t p = 0; p < nphys; ++p) {
        float lo = model.is_log10[p] ? std::log10(los[p]) : los[p];
        float hi = model.is_log10[p] ? std::log10(his[p]) : his[p];
        if(!(nbins[p] >= 1) || !std::isfinite(lo) || !std::isfinite(hi) || !(hi > lo)) {
            log<LOG_ERROR>(L"%1% || Bad grid for parameter %2% (%3%): nbins %4%, internal lo %5%, hi %6%. Terminating.")
                % __func__ % p % model.param_names[p].c_str() % nbins[p] % lo % hi;
            exit(EXIT_FAILURE);
        }
        Eigen::VectorXf ax(nbins[p]);
        if(nbins[p] == 1) {
            ax(0) = 0.5f * (lo + hi);
        } else {
            for(size_t k = 0; k < nbins[p]; ++k)
                ax(k) = lo + (hi - lo) * k / (float)(nbins[p] - 1);
        }
        axes.push_back(ax);
        log<LOG_INFO>(L"%1% || Grid for parameter %2% (%3%): %4% points in [%5%, %6%] (internal units).")
            % __func__ % p % model.param_names[p].c_str() % nbins[p] % lo % hi;
    }

    // Locate the selected channel's bins inside the collapsed truth-variable vector.
    // Collapsed ordering is mode -> detector -> channel, each channel contributing
    // its per-channel bin count for this variable (subchannels already summed).
    // In fixed-baseline mode we also record the selected subchannels' uncollapsed
    // bin ranges so events can be routed into the output histogram directly.
    struct SubchRange { long start, nb, out_off; };
    std::vector<SubchRange> subch_ranges;
    float e_min = 0, e_max = 0;
    size_t offset = 0, out_off = 0;
    for(size_t im = 0; im < config.m_num_modes; ++im) {
        for(size_t id = 0; id < config.m_num_detectors; ++id) {
            for(size_t ic = 0; ic < config.m_num_channels; ++ic) {
                size_t nb = config.GetChannelVariableBins(ic, truth_var).NBins();
                bool selected = config.m_channel_names[ic] == channel_name &&
                    (detector_name.empty() || config.m_detector_names[id] == detector_name);
                if(selected) {
                    truth_blocks.push_back({offset, offset + nb});
                    log<LOG_INFO>(L"%1% || Selected truth block for %2%_%3%_%4%: collapsed bins [%5%, %6%).")
                        % __func__ % config.m_mode_names[im].c_str() % config.m_detector_names[id].c_str()
                        % channel_name.c_str() % offset % (offset + nb);
                    const std::vector<float> &edges = config.GetChannelVariableBins(ic, truth_var).Edges();
                    e_min = truth_e_min = edges.front();
                    e_max = truth_e_max = edges.back();
                    for(size_t sc = 0; sc < config.m_num_subchannels[ic]; ++sc) {
                        std::string subch = config.m_mode_names[im] + "_" + config.m_detector_names[id]
                            + "_" + config.m_channel_names[ic] + "_" + config.m_subchannel_names[ic][sc];
                        long start = config.GetGlobalVariableBinStart(config.GetSubchannelIndex(subch), truth_var);
                        subch_ranges.push_back({start, (long)nb, (long)out_off});
                    }
                    out_off += nb;
                }
                offset += nb;
            }
        }
    }
    if(offset != config.m_num_variable_bins_total_collapsed[truth_var]) {
        log<LOG_ERROR>(L"%1% || Collapsed bin bookkeeping mismatch for variable %2%: counted %3%, expected %4%. Terminating.")
            % __func__ % truth_var % offset % config.m_num_variable_bins_total_collapsed[truth_var];
        exit(EXIT_FAILURE);
    }
    if(truth_blocks.empty()) {
        log<LOG_ERROR>(L"%1% || No truth bins selected for channel %2%, detector '%3%'. Valid channels: %4%, detectors: %5%. Terminating.")
            % __func__ % channel_name.c_str() % detector_name.c_str()
            % config.m_channel_names % config.m_detector_names;
        exit(EXIT_FAILURE);
    }

    if(fixed_baseline >= 0) {
        // Fixed-baseline mode: build an event-weight histogram over (output truth bin,
        // fine truth-energy bin, model rule), so the truth spectrum at any physics
        // point is H_ns * get_probs evaluated at L/E = fixed_baseline / E_fine.
        if(model.ivars.size() != 1) {
            log<LOG_ERROR>(L"%1% || Fixed-baseline mode supports single-ivar (L/E) models only; model has %2% ivars. Terminating.")
                % __func__ % model.ivars.size();
            exit(EXIT_FAILURE);
        }
        const size_t n_fine = 2000;  // fine E grid; probabilities averaged event-weighted within each fine bin
        const size_t J = model.model_functions.size();
        const size_t n_out = out_off;
        H_ns = Eigen::MatrixXf::Zero(n_out, n_fine * J);
        le_fine.assign(1, std::vector<float>(n_fine));
        for(size_t j = 0; j < n_fine; ++j) {
            float e_j = e_min + (j + 0.5f) * (e_max - e_min) / n_fine;
            le_fine[0][j] = fixed_baseline / e_j;
        }
        size_t n_used = 0;
        for(size_t i = 0; i < prop.NEvent(); ++i) {
            int g = prop.VariableBinIndex(truth_var, i);
            if(g < 0) continue;
            const SubchRange *range = nullptr;
            for(const auto &r: subch_ranges)
                if(g >= r.start && g < r.start + r.nb) { range = &r; break; }
            if(!range) continue;
            int r_rule = prop.model_rule[i];
            if(r_rule < 0 || r_rule >= (int)J) continue;
            float e = prop.VariableValue(truth_var, i);
            long j = (long)((e - e_min) / (e_max - e_min) * n_fine);
            if(j < 0 || j >= (long)n_fine) continue;
            H_ns(range->out_off + (g - range->start), r_rule * n_fine + j) += prop.added_weights[i];
            ++n_used;
        }
        log<LOG_INFO>(L"%1% || Fixed-baseline mode: L = %2%, %3% output bins, %4% fine E bins, %5% events used.")
            % __func__ % fixed_baseline % n_out % n_fine % n_used;
        P = FixedBaselineSpectrum(model.default_val);
    } else {
        // CV truth-binned prediction P: model at its default point, nuisances at CV
        // (empty PROsyst, physics-only parameter vector).
        PROspec cv_spec = FillSpectra(config, prop, empty_syst, model, model.default_val, true, truth_var);
        P = SelectTruthBins(CollapseMatrix(config, cv_spec.Spec(), (int)truth_var));
    }
    if(P.sum() <= 0) {
        log<LOG_ERROR>(L"%1% || CV truth-binned prediction for channel %2% sums to %3% (<= 0). Terminating.")
            % __func__ % channel_name.c_str() % P.sum();
        exit(EXIT_FAILURE);
    }
    log<LOG_INFO>(L"%1% || CV truth prediction: %2% selected bins, %3% total events.")
        % __func__ % (int)P.size() % P.sum();
}

Eigen::VectorXf PROnormshape::SelectTruthBins(const Eigen::VectorXf &collapsed) const {
    size_t ntot = 0;
    for(const auto &[start, end]: truth_blocks) ntot += end - start;
    Eigen::VectorXf out(ntot);
    size_t at = 0;
    for(const auto &[start, end]: truth_blocks) {
        out.segment(at, end - start) = collapsed.segment(start, end - start);
        at += end - start;
    }
    return out;
}

Eigen::VectorXf PROnormshape::FixedBaselineSpectrum(const Eigen::VectorXf &phys) const {
    Eigen::MatrixXf probs = metric.GetModel().get_probs(phys, le_fine);
    Eigen::Map<const Eigen::VectorXf> probs_flat(probs.data(), probs.size());
    return H_ns * probs_flat;
}

Eigen::VectorXf PROnormshape::TruthSpectrum(const Eigen::VectorXf &phys) const {
    if(fixed_baseline >= 0)
        return FixedBaselineSpectrum(phys);
    PROspec spec = FillSpectra(config, prop, empty_syst, metric.GetModel(), phys, true, truth_var);
    return SelectTruthBins(CollapseMatrix(config, spec.Spec(), (int)truth_var));
}

void PROnormshape::ComputeNormShape(normShapeOut &out, const Eigen::VectorXf &phys) const {
    Eigen::VectorXf M = TruthSpectrum(phys);
    float sumM = M.sum(), sumP = P.sum();
    out.norm = (sumM - sumP) / sumP;
    out.shape = sumM > 0 ? (M * (sumP / sumM) - P).cwiseAbs().sum() / sumP : 0.0f;
}

std::vector<normShapeOut> PROnormshape::Worker(const PROfitterConfig &fitconfig,
                                               const std::vector<std::vector<float>> &grid,
                                               std::atomic<int> *point_counter, uint32_t seed,
                                               bool statonly, const Eigen::VectorXf &cv_params,
                                               MultiPROgressBar *progress) const {
    std::vector<normShapeOut> outs;
    const int total = (int)grid.size();
    const size_t nphys = metric.GetModel().nparams;

    PROmetric *local_metric = metric.Clone();
    // override_systs keeps a non-owning pointer, so the zero-covariance copy
    // must outlive every metric evaluation in this worker.
    PROsyst statonly_syst;
    if(statonly) {
        statonly_syst = local_metric->GetSysts();
        statonly_syst.fractional_covariance = Eigen::MatrixXf::Constant(
            config.m_num_variable_bins_total[config.i_prime],
            config.m_num_variable_bins_total[config.i_prime], 0);
        local_metric->override_systs(statonly_syst);
    }
    const size_t nsplines = local_metric->GetSysts().GetNSplines();

    while(true) {
        int i = point_counter->fetch_add(1);
        if(i >= total) break;
        local_metric->reset();

        normShapeOut output;
        output.point = grid[i];
        Eigen::VectorXf phys = Eigen::VectorXf::Map(grid[i].data(), nphys);

        // Points violating the model constraint (e.g. unitarity) are recorded as
        // NaN and never evaluated -- probability functions may be undefined there.
        if(local_metric->GetModel().model_constraint &&
           !local_metric->GetModel().model_constraint(phys)) {
            output.chi2 = std::numeric_limits<float>::quiet_NaN();
            output.norm = std::numeric_limits<float>::quiet_NaN();
            output.shape = std::numeric_limits<float>::quiet_NaN();
            outs.push_back(output);
            if(progress) progress->increment_bar(0);
            continue;
        }

        if(statonly || nsplines == 0) {
            // All physics parameters pinned and nothing left to profile:
            // a single metric evaluation with nuisances at CV.
            Eigen::VectorXf params = cv_params;
            params.head(nphys) = phys;
            Eigen::VectorXf empty_grad;
            output.chi2 = (*local_metric)(params, empty_grad, false);
        } else {
            Eigen::VectorXf lb(nphys + nsplines), ub(nphys + nsplines);
            lb << local_metric->GetModel().lb,
                  Eigen::VectorXf::Map(local_metric->GetSysts().spline_lo.data(), nsplines);
            ub << local_metric->GetModel().ub,
                  Eigen::VectorXf::Map(local_metric->GetSysts().spline_hi.data(), nsplines);
            for(size_t si = 0; si < nsplines; ++si) {
                if(!local_metric->GetSysts().spline_has_restrict[si]) continue;
                lb(nphys + si) = local_metric->GetSysts().spline_restrict_lo[si];
                ub(nphys + si) = local_metric->GetSysts().spline_restrict_hi[si];
            }
            // Pin every physics parameter at this grid point; PROfitter profiles the rest.
            lb.head(nphys) = phys;
            ub.head(nphys) = phys;
            local_metric->setBounds(lb, ub);

            PROfitter fitter(ub, lb, fitconfig, seed + (uint32_t)i);
            // Warm start from the nearest already-fitted point of this thread,
            // Euclidean in internal parameter units (same pattern as PROsurf).
            if(!outs.empty()) {
                size_t nearest = 0;
                float best_d2 = std::numeric_limits<float>::max();
                for(size_t k = 0; k < outs.size(); ++k) {
                    if(!outs[k].best_fit.size()) continue;
                    float d2 = 0;
                    for(size_t p = 0; p < nphys; ++p) {
                        const float d = outs[k].point[p] - grid[i][p];
                        d2 += d * d;
                    }
                    if(d2 < best_d2) { best_d2 = d2; nearest = k; }
                }
                output.chi2 = fitter.Fit(*local_metric, outs[nearest].best_fit);
            } else {
                output.chi2 = fitter.Fit(*local_metric, cv_params);
            }
            output.best_fit = fitter.best_fit;
        }

        ComputeNormShape(output, phys);
        outs.push_back(output);
        if(progress) progress->increment_bar(0);
    }

    delete local_metric;
    return outs;
}

void PROnormshape::Scan(const PROfitterConfig &fitconfig, PROseed &proseed, int nthread,
                        bool statonly, const Eigen::VectorXf &cv_params) {
    const size_t nphys = metric.GetModel().nparams;

    size_t total = 1;
    for(size_t p = 0; p < nphys; ++p) total *= nbins[p];
    log<LOG_INFO>(L"%1% || Norm-shape scan: %2% grid points over %3% physics parameters, %4% threads, statonly %5%.")
        % __func__ % total % nphys % nthread % (int)statonly;

    std::vector<std::vector<float>> grid;
    grid.reserve(total);
    for(size_t flat = 0; flat < total; ++flat) {
        std::vector<float> pt(nphys);
        size_t rem = flat;
        for(int p = (int)nphys - 1; p >= 0; --p) {
            pt[p] = axes[p](rem % nbins[p]);
            rem /= nbins[p];
        }
        grid.push_back(std::move(pt));
    }

    std::vector<std::pair<int, std::string>> pb_configs;
    pb_configs.push_back({std::max(1, (int)total), "Norm-shape scan"});
    MultiPROgressBar scan_progress(pb_configs);
    if(fitconfig.progress_bar) {
        scan_progress.initialize_display();
        scan_progress.start_display_thread();
    }
    MultiPROgressBar *pb_ptr = fitconfig.progress_bar ? &scan_progress : nullptr;

    std::atomic<int> point_counter{0};
    std::vector<std::future<std::vector<normShapeOut>>> futures;
    for(int t = 0; t < nthread; ++t) {
        futures.emplace_back(std::async(std::launch::async, [&, t]() {
            return this->Worker(fitconfig, grid, &point_counter,
                                proseed.getThreadSeeds()->at(t), statonly, cv_params, pb_ptr);
        }));
    }
    results.clear();
    for(auto &fut: futures) {
        std::vector<normShapeOut> res = fut.get();
        results.insert(results.end(), res.begin(), res.end());
    }
    if(fitconfig.progress_bar) scan_progress.finish_all();

    // Chi2 at the model default point (all nuisances at CV) as the Asimov floor;
    // the overall min over grid and CV point is subtracted downstream.
    {
        PROmetric *local_metric = metric.Clone();
        PROsyst statonly_syst;
        if(statonly) {
            statonly_syst = local_metric->GetSysts();
            statonly_syst.fractional_covariance = Eigen::MatrixXf::Constant(
                config.m_num_variable_bins_total[config.i_prime],
                config.m_num_variable_bins_total[config.i_prime], 0);
            local_metric->override_systs(statonly_syst);
        }
        Eigen::VectorXf empty_grad;
        Eigen::VectorXf params = cv_params;
        cv_chi2 = (*local_metric)(params, empty_grad, false);
        delete local_metric;
    }
    min_chi2 = cv_chi2;
    for(const auto &r: results)
        if(r.chi2 < min_chi2) min_chi2 = r.chi2;
    log<LOG_INFO>(L"%1% || Scan done: %2% points, cv_chi2 %3%, min_chi2 %4%.")
        % __func__ % results.size() % cv_chi2 % min_chi2;
}

void PROnormshape::Write(const std::string &filename) const {
    const PROmodel &model = metric.GetModel();
    std::ofstream file(filename);
    if(!file) {
        log<LOG_ERROR>(L"%1% || Could not open %2% for writing.") % __func__ % filename.c_str();
        return;
    }
    file << "NormShapeScan v1\n";
    file << "Parameters:";
    for(const auto &name: model.param_names) file << " " << name;
    file << "\nis_log10:";
    for(bool b: model.is_log10) file << " " << (int)b;
    file << "\nGrid:";
    for(size_t n: nbins) file << " " << n;
    file << "\nTruthVar: " << truth_var << "\nChannel: " << channel_name
         << "\nDetector: " << (detector_name.empty() ? "all" : detector_name)
         << "\nBaseline: " << fixed_baseline
         << "\nTruthBins: " << P.size()
         << "\nCVchi2: " << cv_chi2 << "\nMinChi2: " << min_chi2 << "\n\n";
    for(const auto &name: model.param_names) file << name << " ";
    file << "chi2 dchi2 norm shape";
    for(const auto &r: results) {
        file << "\n";
        for(float v: r.point) file << v << " ";
        file << r.chi2 << " " << (r.chi2 - min_chi2) << " " << r.norm << " " << r.shape;
    }
    file << "\n";
    log<LOG_INFO>(L"%1% || Wrote %2% grid points to %3%.") % __func__ % results.size() % filename.c_str();
}

}
