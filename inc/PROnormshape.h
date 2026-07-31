/**
 * @file PROnormshape.h
 * @brief N-dimensional physics-parameter grid scan producing (norm, shape) coordinates.
 * @author PROfit Collaboration
 *
 * @details Defines PROnormshape: a scan over a cartesian grid covering ALL physics
 * parameters of a model. At each grid point it records
 *   - a chi-squared vs the data, profiled over nuisance splines (or a direct
 *     stat-only metric evaluation),
 *   - the (norm, shape) coordinates of the truth-binned prediction M relative to
 *     the CV (model default) truth-binned prediction P:
 *         norm  = (sum_i M_i - sum_i P_i) / sum_i P_i
 *         shape = sum_i |M_i * (sum_j P_j / sum_k M_k) - P_i| / sum_i P_i
 * The truth bins are the collapsed bins of one named channel for a chosen
 * (truth-level) variable, concatenated side-by-side across all modes and
 * detectors. Side-by-side (rather than summed) means a normalization-only
 * change in a single detector still registers as a shape difference.
 *
 * Downstream, the (shape, norm) plane is binned and profiled (min chi2 per bin)
 * to draw exclusion contours in norm-shape space; see plotting scripts.
 */
#ifndef PRONORMSHAPE_H
#define PRONORMSHAPE_H

#include "PROfitter.h"
#include "PROconfig.h"
#include "PROpeller.h"
#include "PROsyst.h"
#include "PROseed.h"
#include "PROmetric.h"
#include "PROgress.h"

#include <Eigen/Eigen>

#include <atomic>
#include <string>
#include <vector>

namespace PROfit {

    /**
     * @brief Output record for a single grid point of a norm-shape scan.
     */
    struct normShapeOut {
        std::vector<float> point;  ///< Physics parameter values in model-internal units (log10 where is_log10).
        float chi2 = -1;           ///< Raw chi-squared (not min-subtracted), profiled over splines unless stat-only.
        float norm = 0;            ///< Normalization change of the truth-binned prediction relative to CV.
        float shape = 0;           ///< Shape change of the truth-binned prediction relative to CV.
        Eigen::VectorXf best_fit;  ///< Full best-fit vector (physics + splines); empty in stat-only mode.
    };

    /**
     * @brief Grid scan over all physics parameters with chi2 + (norm, shape) per point.
     */
    class PROnormshape {
        public:
            PROmetric &metric;         ///< Metric providing the chi2, model, and systematics.
            const PROconfig &config;   ///< Configuration (binning, channels, collapsing matrices).
            const PROpeller &prop;     ///< MC events, needed for FillSpectra.
            size_t truth_var;          ///< Variable index of the truth binning used for norm/shape.
            std::string channel_name;  ///< Channel whose collapsed truth bins enter norm/shape.
            std::string detector_name; ///< Restrict truth bins to this detector ("" = all detectors).
            /// Fixed oscillation baseline in the L/E units of the config (e.g. km for L/E in
            /// km/GeV). If >= 0, oscillation probabilities for the norm/shape spectra are
            /// evaluated at L/E = fixed_baseline / E_true instead of each event's own true L/E,
            /// so the coordinates become a detector-independent, common definition.
            float fixed_baseline = -1;

            std::vector<size_t> nbins;             ///< Grid points per physics parameter.
            std::vector<Eigen::VectorXf> axes;     ///< Grid values per parameter, model-internal units.
            /// [start, end) ranges of the selected channel in the collapsed truth-variable
            /// vector, one per (mode, detector) block, in collapsed-vector order.
            std::vector<std::pair<size_t, size_t>> truth_blocks;
            Eigen::VectorXf P;   ///< CV truth-binned prediction over the selected bins (concatenated).

            std::vector<normShapeOut> results;  ///< One entry per grid point after Scan().
            float cv_chi2 = -1;   ///< Metric evaluated at the model default point (all nuisances at CV).
            float min_chi2 = -1;  ///< min(cv_chi2, min over grid) after Scan().

            /**
             * @brief Set up the grid and the CV truth-binned prediction.
             * @param los,his Per-parameter grid limits in LINEAR units; converted to
             *        model-internal (log10) units where the model flags is_log10.
             *        Grids are uniform in internal units (log-uniform for log10 params)
             *        and include both endpoints.
             */
            PROnormshape(PROmetric &metric, const PROconfig &config, const PROpeller &prop,
                         size_t truth_var, const std::string &channel_name,
                         const std::vector<size_t> &nbins,
                         const std::vector<float> &los, const std::vector<float> &his,
                         const std::string &detector_name = "", float fixed_baseline = -1);

            /**
             * @brief Run the scan over the full grid with nthread workers.
             * @param statonly If true, zero out the fractional covariance and evaluate the
             *        metric directly with nuisances at CV (no fits). Otherwise every grid
             *        point pins all physics parameters and profiles the splines with PROfitter.
             * @param cv_params Full CV parameter vector (physics defaults + spline CVs),
             *        sized nparams + GetNSplines() of the metric's systematics.
             */
            void Scan(const PROfitterConfig &fitconfig, PROseed &proseed, int nthread,
                      bool statonly, const Eigen::VectorXf &cv_params);

            /** @brief Write the scan results as a whitespace-separated text file with header. */
            void Write(const std::string &filename) const;

            /** @brief Compute (norm, shape) for a physics point (internal units) against P. */
            void ComputeNormShape(normShapeOut &out, const Eigen::VectorXf &phys) const;

            /** @brief Truth spectrum M for a physics point (internal units), same bins as P. */
            Eigen::VectorXf TruthSpectrum(const Eigen::VectorXf &phys) const;

            float truth_e_min = 0;  ///< Lower edge of the selected channel's truth binning.
            float truth_e_max = 0;  ///< Upper edge of the selected channel's truth binning.

        private:
            PROsyst empty_syst;  ///< Empty systematics: truth spectra are evaluated with nuisances at CV.

            /// Fixed-baseline mode: event-weight histogram of shape
            /// (n_out_bins, n_fine * n_model_functions), column index = rule * n_fine + fine_bin,
            /// mirroring PROmodel::H_combined but with the physics grid replaced by a fine
            /// truth-energy grid at the fixed baseline.
            Eigen::MatrixXf H_ns;
            std::vector<std::vector<float>> le_fine;  ///< {L/E values at fine-bin centers}, input to get_probs.

            /** @brief Gather the selected channel's bins from a collapsed truth-variable vector. */
            Eigen::VectorXf SelectTruthBins(const Eigen::VectorXf &collapsed) const;

            /** @brief Fixed-baseline truth spectrum: H_ns * probs(phys) at L/E = baseline / E. */
            Eigen::VectorXf FixedBaselineSpectrum(const Eigen::VectorXf &phys) const;

            /** @brief Thread worker: dynamic dispatch over grid points via shared counter. */
            std::vector<normShapeOut> Worker(const PROfitterConfig &fitconfig,
                                             const std::vector<std::vector<float>> &grid,
                                             std::atomic<int> *point_counter, uint32_t seed,
                                             bool statonly, const Eigen::VectorXf &cv_params,
                                             MultiPROgressBar *progress) const;
    };

}

#endif
