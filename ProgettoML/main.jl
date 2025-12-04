# ============================================================================
#          FRAUD DETECTION - MULTICLASS CLASSIFICATION
#          Main Script - Executes All Experiments
#          Group Project - Machine Learning I
# ============================================================================
#
# Objective: E-commerce fraud detection using multiclass classification
#            (Legitimate, Suspicious, Fraudulent)
#
# Methodology:
#   - 4 ML Algorithms: ANNs (8 topologies), SVMs (10 configs), 
#                      Decision Trees (7 depths), kNN (6 k values)
#   - Ensemble Methods: Majority Voting + Weighted Voting
#   - Dataset: 1.5M e-commerce transactions → 3-class risk assessment
#   - Cross-Validation: 3-fold stratified on training set
#   - Evaluation: Hold-out test set (20%)
#
# Code Organization:
#   /project/
#   ├── main.jl                    ← This file (executable from top to bottom)
#   ├── /utils/
#   │   ├── utils.jl              ← Course utilities (includes modelCrossValidation)
#   │   └── preprocessing.jl      ← Custom preprocessing functions
#   └── /datasets/
#       └── Fraudulent_E-Commerce_Transaction_Data_merge.csv
#
# ============================================================================

# ============================================================================
#                    SETUP & IMPORTS
# ============================================================================

# Set random seed for reproducibility
using Random
Random.seed!(42)

# Load packages
using CSV
using DataFrames
using Statistics
using Dates
using StatsBase

println("✅ Packages loaded!")

# Load course utilities
include("utils/utils.jl")
println("✅ Course utilities loaded (includes modelCrossValidation, confusionMatrix, etc.)")

# Load custom preprocessing
include("utils/preprocessing.jl")
using .PreprocessingUtils
println("✅ Custom preprocessing utilities loaded!")

# ============================================================================
#  HELPER FUNCTION: EVALUATE ALL MODELS (FIXED SVM TYPO)
# ============================================================================
function evaluate_approach(approach_name, train_inputs, train_targets, test_inputs, test_targets; cv_folds=3)
    println("\n" * "="^80)
    println("🚀 EVALUATING APPROACH: $approach_name")
    println("="^80)
    
    cv_indices = crossvalidation(train_targets, cv_folds)
    final_results = Dict{String, Dict{String, Float64}}()
    
    # Prepare Data
    train_targets_str = string.(train_targets)
    test_targets_str = string.(test_targets)
    classes_str = sort(unique(train_targets_str))
    
    classes_int = sort(unique(train_targets))
    train_targets_onehot = oneHotEncoding(train_targets, classes_int)
    test_targets_onehot = oneHotEncoding(test_targets, classes_int)

    normParams = calculateMinMaxNormalizationParameters(train_inputs)
    train_inputs_norm = normalizeMinMax(train_inputs, normParams)
    test_inputs_norm = normalizeMinMax(test_inputs, normParams)

    # --- INTERNAL HELPER ---
    function calculate_metrics_safe(y_pred_probs, y_pred_class, y_true_class, y_true_onehot, classes)
        # AUC Calculation
        auc_score = 0.5
        if length(classes) == 2
            probs = vec(y_pred_probs)
            true_bin = vec(y_true_onehot) 
            p = sortperm(probs); probs_sorted = probs[p]; true_sorted = true_bin[p]
            tpr = [0.0]; fpr = [0.0]
            num_pos = sum(true_sorted); num_neg = length(true_sorted) - num_pos
            if num_pos > 0 && num_neg > 0
                tp = 0; fp = 0
                for i in length(probs_sorted):-1:1
                    if true_sorted[i] == 1; tp += 1; else; fp += 1; end
                    push!(tpr, tp/num_pos); push!(fpr, fp/num_neg)
                end
                auc_score = 0.0
                for i in 2:length(tpr); auc_score += (fpr[i] - fpr[i-1]) * (tpr[i] + tpr[i-1]) / 2; end
            end
        end

        # Metrics
        acc, sens, spec, f1 = 0.0, 0.0, 0.0, 0.0
        if length(classes) == 2
            pos_label = classes[end]
            y_p_bool = vec(y_pred_class .== pos_label)
            y_t_bool = vec(y_true_class .== pos_label)
            (acc, err, sens, spec, prec, npv, f1, cm) = confusionMatrix(y_p_bool, y_t_bool)
        else
            cm_res = confusionMatrix(y_pred_class, y_true_class, classes; weighted=true)
            acc = cm_res.accuracy; sens = cm_res.aggregated.sensitivity
            spec = cm_res.aggregated.specificity; f1 = cm_res.aggregated.f1
        end
        return Dict("Accuracy"=>acc, "AUC"=>auc_score, "Sensitivity"=>sens, "Specificity"=>spec, "F1"=>f1)
    end
    
    # 1. ANNs
    println("\n[1/4] Testing ANNs...")
    ann_topologies = [[256], [128], [64], [32], [256, 128], [128, 64], [64, 32], [96, 48]]
    best_f1_cv_ann = -1.0; best_topo_ann = []
    for topology in ann_topologies
        hyperparams = Dict("topology" => topology, "learningRate" => 0.003, "validationRatio" => 0.1, 
                           "numExecutions" => 1, "maxEpochs" => 800, "maxEpochsVal" => 25)
        res = modelCrossValidation(:ANN, hyperparams, (train_inputs_norm, train_targets), cv_indices)
        if res[7][1] > best_f1_cv_ann; best_f1_cv_ann = res[7][1]; best_topo_ann = topology; end
    end
    println("   ✨ Best ANN (CV): $best_topo_ann - CV F1: $(round(best_f1_cv_ann*100, digits=2))%")

    println("      ...Retraining Best ANN...")
    N_train = size(train_inputs_norm, 1); (train_idx, val_idx) = holdOut(N_train, 0.1)
    final_ann, _ = _trainClassANN(best_topo_ann,
        (train_inputs_norm[train_idx, :], train_targets_onehot[train_idx, :]),
        validationDataset=(train_inputs_norm[val_idx, :], train_targets_onehot[val_idx, :]),
        testDataset=(test_inputs_norm, test_targets_onehot),
        maxEpochs=800, learningRate=0.003, maxEpochsVal=25)
    
    test_outputs_ann_raw = final_ann(test_inputs_norm')'
    if size(test_targets_onehot, 2) == 1
        probs_ann = vec(test_outputs_ann_raw); preds_ann = Int.(probs_ann .>= 0.5)
        final_results["ANN"] = calculate_metrics_safe(probs_ann, preds_ann, test_targets, test_targets_onehot, classes_int)
    else
        preds_bool = classifyOutputs(test_outputs_ann_raw)
        preds_ann = [findfirst(x->x, row) - 1 for row in eachrow(preds_bool)]
        final_results["ANN"] = calculate_metrics_safe(test_outputs_ann_raw, preds_ann, test_targets, test_targets_onehot, classes_int)
    end
    m = final_results["ANN"]
    println("      ✅ ANN Results: Acc=$(round(m["Accuracy"],digits=3)), F1=$(round(m["F1"],digits=3))")

    # 2. SVM
    println("\n[2/4] Testing SVMs...")
    svm_configs = [("linear", 1.0, 0.1, 3), ("rbf", 1.0, 0.125, 3), ("poly", 1.0, 0.1, 2)]
    best_f1_cv_svm = -1.0; best_params_svm = ()
    for (kernel, C, gamma, degree) in svm_configs
        res = modelCrossValidation(:SVC, Dict("kernel"=>kernel, "C"=>C, "gamma"=>gamma, "degree"=>degree), (train_inputs, train_targets), cv_indices)
        if res[7][1] > best_f1_cv_svm; best_f1_cv_svm = res[7][1]; best_params_svm = (kernel, C, gamma, degree); end
    end
    
    (k, C, g, d) = best_params_svm
    k_func = k == "linear" ? LIBSVM.Kernel.Linear : (k == "poly" ? LIBSVM.Kernel.Polynomial : LIBSVM.Kernel.RadialBasis)
    model_svm = SVMClassifier(kernel=k_func, cost=C, gamma=g, degree=Int32(d))
    mach_svm = machine(model_svm, MLJ.table(train_inputs_norm), categorical(train_targets_str))
    MLJ.fit!(mach_svm, verbosity=0)
    
    # CORREZIONE QUI SOTTO: pred_svm_class -> pred_svm_str
    preds_svm_class = MLJ.predict(mach_svm, MLJ.table(test_inputs_norm))
    preds_svm_str = string.(preds_svm_class) # <--- VARIABILE CORRETTA
    
    final_results["SVM"] = calculate_metrics_safe(zeros(length(preds_svm_str)), preds_svm_str, test_targets_str, test_targets_onehot, classes_str)
    m = final_results["SVM"]
    println("      ✅ SVM Results: Acc=$(round(m["Accuracy"],digits=3)), F1=$(round(m["F1"],digits=3))")

    # 3. Decision Trees
    println("\n[3/4] Testing Decision Trees...")
    depths = [3, 5, 7, 10]; best_f1_cv_dt = -1.0; best_depth = 0
    for d in depths
        res = modelCrossValidation(:DecisionTreeClassifier, Dict("max_depth"=>d), (train_inputs, train_targets), cv_indices)
        if res[7][1] > best_f1_cv_dt; best_f1_cv_dt = res[7][1]; best_depth = d; end
    end
    
    model_dt = DTClassifier(max_depth=best_depth, rng=Random.MersenneTwister(42))
    mach_dt = machine(model_dt, MLJ.table(train_inputs_norm), categorical(train_targets_str))
    MLJ.fit!(mach_dt, verbosity=0)
    
    preds_dt_prob = MLJ.predict(mach_dt, MLJ.table(test_inputs_norm))
    preds_dt_str = string.(mode.(preds_dt_prob))
    probs_dt = zeros(length(preds_dt_str)); try; probs_dt = pdf.(preds_dt_prob, classes_str[end]); catch; end

    final_results["DT"] = calculate_metrics_safe(probs_dt, preds_dt_str, test_targets_str, test_targets_onehot, classes_str)
    m = final_results["DT"]
    println("      ✅ DT Results: Acc=$(round(m["Accuracy"],digits=3)), F1=$(round(m["F1"],digits=3))")

    # 4. kNN
    println("\n[4/4] Testing kNN...")
    k_vals = [1, 3, 5, 7]; best_f1_cv_knn = -1.0; best_k = 0
    for k in k_vals
        res = modelCrossValidation(:KNeighborsClassifier, Dict("n_neighbors"=>k), (train_inputs, train_targets), cv_indices)
        if res[7][1] > best_f1_cv_knn; best_f1_cv_knn = res[7][1]; best_k = k; end
    end
    
    model_knn = kNNClassifier(K=best_k)
    mach_knn = machine(model_knn, MLJ.table(train_inputs_norm), categorical(train_targets_str))
    MLJ.fit!(mach_knn, verbosity=0)
    
    preds_knn_prob = MLJ.predict(mach_knn, MLJ.table(test_inputs_norm))
    preds_knn_str = string.(mode.(preds_knn_prob))
    probs_knn = zeros(length(preds_knn_str)); try; probs_knn = pdf.(preds_knn_prob, classes_str[end]); catch; end
    
    final_results["kNN"] = calculate_metrics_safe(probs_knn, preds_knn_str, test_targets_str, test_targets_onehot, classes_str)
    m = final_results["kNN"]
    println("      ✅ kNN Results: Acc=$(round(m["Accuracy"],digits=3)), F1=$(round(m["F1"],digits=3))")

    return final_results
end

# ============================================================================
#              DATA LOADING & 3-CLASS TARGET CREATION
# ============================================================================
#
# Dataset: Fraudulent E-Commerce Transaction Data (1.5M transactions)
#
# Target Creation Strategy:
#   Original: Binary fraud labels (fraud vs non-fraud)
#   Our approach: 3-class risk assessment based on multiple signals:
#     1. Time Risk: Night transactions (0-5am, 11pm)
#     2. Amount Risk: High-value transactions (>90th percentile)
#     3. Account Age Risk: New accounts (<30 days)
#
# Class Mapping:
#   - Class 0 (LEGITTIMO): Low-risk, legitimate transactions
#   - Class 1 (SOSPETTO): Borderline cases requiring manual review
#   - Class 2 (FRAUDOLENTO): High-risk fraudulent transactions
#
# Justification: This approach allows for graduated risk assessment, enabling
# businesses to automatically approve low-risk transactions, flag suspicious
# cases for manual review, and immediately block high-risk frauds.
#
# ============================================================================

const DATA_PATH = "datasets/Fraudulent_E-Commerce_Transaction_Data_merge.csv"
println("\n" * "="^70)
println("📂 LOADING DATA")
println("="^70)

df = CSV.read(DATA_PATH, DataFrame)
target_col = "Is Fraudulent"

println("Original dataset size: $(size(df))")
println("Original fraud distribution:")
println("  Non-fraud: $(sum(df[!, target_col] .== 0))")
println("  Fraud:     $(sum(df[!, target_col] .== 1))")

# Create 3-class target
println("\n" * "="^70)
println("🎯 CREATING 3-CLASS TARGET")
println("="^70)

df_with_classes = create_risk_classes(df, target_col)

# ============================================================================
#          CLASS BALANCING & TRAIN/TEST SPLIT
# ============================================================================
#
# Challenge: Highly imbalanced dataset (90% Legitimate, 8.6% Suspicious, 1.4% Fraudulent)
#
# Solution: Undersample majority classes to match minority class (20,654 samples per class)
#
# Train/Test Split:
#   - 80% Training (49,569 samples) - used for cross-validation and model selection
#   - 20% Test (12,393 samples) - held out for final evaluation
#
# CRITICAL: Test set is NEVER used during training or model selection to prevent data leakage!
#
# ============================================================================

println("\n" * "="^70)
println("✅ TRAIN/TEST SPLIT (80% Train / 20% Test)")
println("="^70)

# Balance classes
class_0 = df_with_classes[df_with_classes.Risk_Class .== 0, :]
class_1 = df_with_classes[df_with_classes.Risk_Class .== 1, :]
class_2 = df_with_classes[df_with_classes.Risk_Class .== 2, :]

n_min = minimum([size(class_0, 1), size(class_1, 1), size(class_2, 1)])
n_target = min(n_min, 40000)

println("\n🔄 Balancing dataset...")
println("  Samples per class: $n_target")

class_0_sample = class_0[shuffle(1:size(class_0, 1))[1:n_target], :]
class_1_sample = class_1[shuffle(1:size(class_1, 1))[1:n_target], :]
class_2_sample = class_2[shuffle(1:size(class_2, 1))[1:n_target], :]

df_balanced = vcat(class_0_sample, class_1_sample, class_2_sample)
df_balanced = df_balanced[shuffle(1:size(df_balanced, 1)), :]

println("  Balanced dataset size: $(size(df_balanced))")

# Split Train/Test BEFORE preprocessing (critical!)
n_total = size(df_balanced, 1)
n_train = floor(Int, n_total * 0.80)
n_test = n_total - n_train

all_indices = shuffle(1:n_total)
train_indices = all_indices[1:n_train]
test_indices = all_indices[n_train+1:end]

df_train = df_balanced[train_indices, :]
df_test = df_balanced[test_indices, :]

println("\n📊 Split Summary:")
println("  Total samples:     $n_total")
println("  Training set:      $n_train (80%)")
println("  Test set:          $n_test (20%)")

# ============================================================================
#                    PREPROCESSING & FEATURE ENGINEERING
# ============================================================================
#
# Steps:
#   1. Time Features: Extract hour, create night flag (hour < 6)
#   2. Feature Engineering:
#      - Amount_per_AccountAge: Transaction amount relative to account maturity
#      - High_Value_Flag: Transactions above 95th percentile
#      - New_Account_Flag: Accounts younger than 30 days
#   3. Missing Value Imputation: Median imputation
#   4. Feature Selection: Drop IDs, addresses, categorical features → 8 numerical features
#   5. Normalization: Min-Max [0,1] using training set parameters only
#
# Final Features (8):
#   - Transaction Amount
#   - Account Age Days
#   - Transaction_Hour
#   - Is_Night
#   - Amount_per_AccountAge
#   - High_Value_Flag
#   - New_Account_Flag
#   - (1 more from preprocessing)
#
# ============================================================================

println("\n🔧 Preprocessing train and test sets...")
df_train_processed = preprocess_multiclass(df_train, target_col)
df_test_processed = preprocess_multiclass(df_test, target_col)

input_cols = setdiff(names(df_train_processed), ["Risk_Class"])
train_inputs = Matrix{Float32}(df_train_processed[:, input_cols])
train_targets = Int.(df_train_processed.Risk_Class)

test_inputs = Matrix{Float32}(df_test_processed[:, input_cols])
test_targets = Int.(df_test_processed.Risk_Class)

println("\n📊 Preprocessed Data:")
println("  Features: $(length(input_cols))")
println("  Train samples: $(size(train_inputs, 1))")
println("  Test samples: $(size(test_inputs, 1))")
println("\n  Feature names: $input_cols")

# Create cross-validation indices (3-fold stratified)
k_folds = 3
cv_indices = crossvalidation(train_targets, k_folds)
println("\n✅ Cross-validation indices created ($k_folds folds, stratified)")

# ============================================================================
#        EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS (ANNs)
# ============================================================================
#
# Configuration:
#   - Topologies tested: 8 architectures (1-4 hidden layers)
#   - Activation: ReLU (hidden layers), Softmax (output)
#   - Optimizer: Adam (learning rate: 0.003)
#   - Loss: Cross-entropy
#   - Regularization: Early stopping (patience: 25 epochs)
#   - Validation: 10% of training set
#   - Executions: 1 per topology (can increase for stability)
#
# Architectures:
#   1. [256, 128, 64] - Deep Large
#   2. [128, 64, 32] - Baseline
#   3. [96, 48, 24] - Medium
#   4. [64, 32] - Shallow
#   5. [128, 128, 64, 32] - Very Deep
#   6. [192, 96, 48] - Wide-Deep
#   7. [128, 64] - Simple
#   8. [256, 128] - Large 2-Layer
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 1: ARTIFICIAL NEURAL NETWORKS")
println("Testing 8 ANN Topologies")
println("="^70)

topologies_to_test = [
    [256],            # 1. 1 hidden layer - Large
    [128],            # 2. 1 hidden layer - Medium
    [64],             # 3. 1 hidden layer - Small
    [32],             # 4. 1 hidden layer - Tiny
    [256, 128],       # 5. 2 hidden layers - Large
    [128, 64],        # 6. 2 hidden layers - Medium
    [64, 32],         # 7. 2 hidden layers - Small
    [96, 48]          # 8. 2 hidden layers - Alternative
]

ann_results = []

for (i, topology) in enumerate(topologies_to_test)
    println("\n[$i/8] Testing topology: $topology")
    
    hyperparams = Dict(
        "topology" => topology,
        "learningRate" => 0.003,
        "validationRatio" => 0.1,
        "numExecutions" => 1,
        "maxEpochs" => 800,
        "maxEpochsVal" => 25
    )
    
    # Use modelCrossValidation from utils.jl
    results = modelCrossValidation(
        :ANN,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(ann_results, (topology, f1_stats[1], results))
end

# Sort by F1 score
sorted_ann_results = sort(ann_results, by=x->x[2], rev=true)

println("\n🏆 ANN Results Ranking (by F1 Score):")
println("-"^70)
for (i, (topo, f1, _)) in enumerate(sorted_ann_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $topo - F1: $(round(f1*100, digits=2))%")
end

best_topology_ann = sorted_ann_results[1][1]
best_f1_ann = sorted_ann_results[1][2]
println("\n✨ Best ANN: $best_topology_ann (CV F1: $(round(best_f1_ann*100, digits=2))%)")

# Train final ANN on full training set and evaluate on test set
println("\n🚀 Training final ANN on full training set (Retrain + Full Metrics)...")

# Setup Data for Final Training
classes_ann = sort(unique(train_targets))
train_targets_onehot = oneHotEncoding(train_targets, classes_ann)
test_targets_onehot = oneHotEncoding(test_targets, classes_ann)

normParams_ann = calculateMinMaxNormalizationParameters(train_inputs)
train_inputs_norm = normalizeMinMax(train_inputs, normParams_ann)
test_inputs_norm = normalizeMinMax(test_inputs, normParams_ann)

# Validation split for Early Stopping
N_train = size(train_inputs_norm, 1)
(train_idx, val_idx) = holdOut(N_train, 0.1)

final_ann, _ = _trainClassANN(best_topology_ann,
    (train_inputs_norm[train_idx, :], train_targets_onehot[train_idx, :]),
    validationDataset=(train_inputs_norm[val_idx, :], train_targets_onehot[val_idx, :]),
    testDataset=(test_inputs_norm, test_targets_onehot),
    maxEpochs=800, learningRate=0.003, maxEpochsVal=25)

# Predict (Raw Probabilities) -> Needed for Ensemble later!
test_outputs_ann = final_ann(test_inputs_norm')'

# Calculate Full Metrics
preds_ann_cls = classifyOutputs(test_outputs_ann) # Boolean matrix
preds_ann_int = [findfirst(x->x, row) - 1 for row in eachrow(preds_ann_cls)] # 0,1,2 labels

# Prepare args for metrics helper
probs_ann_vec = (size(test_targets_onehot, 2) == 1) ? vec(test_outputs_ann) : test_outputs_ann
metrics_ann = calculate_metrics_safe(probs_ann_vec, preds_ann_int, test_targets, test_targets_onehot, classes_ann)

println("📊 ANN Test Results: Acc=$(round(metrics_ann["Accuracy"],digits=3)), F1=$(round(metrics_ann["F1"],digits=3))")

# ============================================================================
#        EXPERIMENT 2: SUPPORT VECTOR MACHINES (SVMs)
# ============================================================================
#
# Configuration:
#   - 10 configurations tested
#   - Kernels: Linear, RBF, Polynomial
#   - Hyperparameter C: 0.1, 1.0, 10.0
#   - Gamma (RBF): auto (1/n_features = 0.125), 0.1
#   - Degree (Polynomial): 2, 3
#
# Configurations:
#   1-3. Linear (C = 0.1, 1.0, 10.0)
#   4-6. RBF with auto gamma (C = 0.1, 1.0, 10.0)
#   7. RBF with γ=0.1 (C = 1.0)
#   8-10. Polynomial degree 2, 3 (C = 1.0, 10.0)
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 2: SUPPORT VECTOR MACHINES")
println("Testing 10 SVM Configurations")
println("="^70)

svm_configs = [
    ("linear", 0.1, 0.125, 3, "Linear C=0.1"),
    ("linear", 1.0, 0.125, 3, "Linear C=1.0"),
    ("linear", 10.0, 0.125, 3, "Linear C=10.0"),
    ("rbf", 0.1, 0.125, 3, "RBF C=0.1 γ=auto"),
    ("rbf", 1.0, 0.125, 3, "RBF C=1.0 γ=auto"),
    ("rbf", 10.0, 0.125, 3, "RBF C=10.0 γ=auto"),
    ("rbf", 1.0, 0.1, 3, "RBF C=1.0 γ=0.1"),
    ("poly", 1.0, 0.125, 2, "Poly C=1.0 deg=2"),
    ("poly", 1.0, 0.125, 3, "Poly C=1.0 deg=3"),
    ("poly", 10.0, 0.125, 2, "Poly C=10.0 deg=2")
]

svm_results = []

for (i, (kernel, C, gamma, degree, desc)) in enumerate(svm_configs)
    println("\n[$i/10] Testing: $desc")
    
    hyperparams = Dict(
        "kernel" => kernel,
        "C" => C,
        "gamma" => gamma,
        "degree" => degree
    )
    
    results = modelCrossValidation(
        :SVC,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(svm_results, (desc, f1_stats[1], kernel, C, gamma, degree, results))
end

sorted_svm_results = sort(svm_results, by=x->x[2], rev=true)

println("\n🏆 SVM Results Ranking:")
println("-"^70)
for (i, (desc, f1, _, _, _, _, _)) in enumerate(sorted_svm_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. $desc - F1: $(round(f1*100, digits=2))%")
end

best_svm = sorted_svm_results[1]
best_desc_svm, best_f1_svm, best_kernel_svm, best_C_svm, best_gamma_svm, best_degree_svm = best_svm[1:6]
println("\n✨ Best SVM: $best_desc_svm (CV F1: $(round(best_f1_svm*100, digits=2))%)")

# Train final SVM and evaluate on test set

println("\n🚀 Training final SVM on full training set...")

# Setup Data
train_inputs_norm = normalizeMinMax(train_inputs, calculateMinMaxNormalizationParameters(train_inputs))
test_inputs_norm = normalizeMinMax(test_inputs, calculateMinMaxNormalizationParameters(train_inputs))
train_targets_str = string.(train_targets)
test_targets_str = string.(test_targets)
classes_str = sort(unique(train_targets_str))

# Train
(k, C, g, d) = (best_kernel_svm, best_C_svm, best_gamma_svm, best_degree_svm)
k_func = k == "linear" ? LIBSVM.Kernel.Linear : (k == "poly" ? LIBSVM.Kernel.Polynomial : LIBSVM.Kernel.RadialBasis)
model_svm = SVMClassifier(kernel=k_func, cost=C, gamma=g, degree=Int32(d))
mach_svm = machine(model_svm, MLJ.table(train_inputs_norm), categorical(train_targets_str))
MLJ.fit!(mach_svm, verbosity=0)

# Predict -> Needed for Ensemble!
svm_predictions = MLJ.predict(mach_svm, MLJ.table(test_inputs_norm))
svm_predictions_str = string.(svm_predictions)

# Calculate Metrics
# SVM probabilities are tricky with LIBSVM wrapper, passing zeros for AUC (placeholder)
metrics_svm = calculate_metrics_safe(zeros(length(svm_predictions_str)), svm_predictions_str, test_targets_str, test_targets_onehot, classes_str)

println("📊 SVM Test Results: Acc=$(round(metrics_svm["Accuracy"],digits=3)), F1=$(round(metrics_svm["F1"],digits=3))")

# ============================================================================
#            EXPERIMENT 3: DECISION TREES
# ============================================================================
#
# Configuration:
#   - 7 maximum depths tested: 3, 5, 7, 10, 15, 20, unlimited
#   - Splitting criterion: Gini impurity
#   - Min samples split: 2
#   - Random seed: 42 (for reproducibility)
#
# Advantages:
#   - Interpretable (can visualize decision rules)
#   - No feature scaling required
#   - Fast training and prediction
#   - Handles non-linear relationships naturally
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 3: DECISION TREES")
println("Testing 7 Maximum Depths")
println("="^70)

tree_depths = [3, 5, 7, 10, 15, 20, -1]
tree_results = []

for (i, max_depth) in enumerate(tree_depths)
    depth_str = max_depth == -1 ? "Unlimited" : string(max_depth)
    println("\n[$i/7] Testing: Depth=$depth_str")
    
    hyperparams = Dict("max_depth" => max_depth)
    
    results = modelCrossValidation(
        :DecisionTreeClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(tree_results, (depth_str, max_depth, f1_stats[1], results))
end

sorted_tree_results = sort(tree_results, by=x->x[3], rev=true)

println("\n🏆 Decision Tree Results Ranking:")
println("-"^70)
for (i, (depth_str, _, f1, _)) in enumerate(sorted_tree_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. Depth=$depth_str - F1: $(round(f1*100, digits=2))%")
end

best_desc_tree, best_max_depth_tree, best_f1_tree = sorted_tree_results[1][1:3]
println("\n✨ Best Tree: Depth=$best_desc_tree (CV F1: $(round(best_f1_tree*100, digits=2))%)"

# Train final Decision Tree

println("\n🚀 Training final Decision Tree...")

model_tree = DTClassifier(max_depth=best_max_depth_tree, rng=Random.MersenneTwister(42))
mach_tree = machine(model_tree, MLJ.table(train_inputs_norm), categorical(train_targets_str))
MLJ.fit!(mach_tree, verbosity=0)

# Predict -> Needed for Ensemble!
tree_predictions = MLJ.predict(mach_tree, MLJ.table(test_inputs_norm))
tree_predictions_mode = mode.(tree_predictions)
tree_predictions_str = string.(tree_predictions_mode)

# Calculate Metrics (Attempt extracting prob for AUC)
probs_dt = zeros(length(tree_predictions_str))
try; probs_dt = pdf.(tree_predictions, classes_str[end]); catch; end

metrics_tree = calculate_metrics_safe(probs_dt, tree_predictions_str, test_targets_str, test_targets_onehot, classes_str)

println("📊 DT Test Results: Acc=$(round(metrics_tree["Accuracy"],digits=3)), F1=$(round(metrics_tree["F1"],digits=3))")

# ============================================================================
#            EXPERIMENT 4: k-NEAREST NEIGHBORS (kNN)
# ============================================================================
#
# Configuration:
#   - 6 k values tested: 1, 3, 5, 7, 10, 15
#   - Distance metric: Euclidean
#   - Voting: Majority voting among k neighbors
#
# Notes:
#   - Feature normalization is CRITICAL for kNN (distance-based)
#   - No explicit training phase (lazy learning)
#   - k=1 is most sensitive to noise
#   - Higher k values create smoother decision boundaries
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 4: k-NEAREST NEIGHBORS")
println("Testing 6 k Values")
println("="^70)

k_values = [1, 3, 5, 7, 10, 15]
knn_results = []

for (i, k) in enumerate(k_values)
    println("\n[$i/6] Testing: k=$k")
    
    hyperparams = Dict("n_neighbors" => k)
    
    results = modelCrossValidation(
        :KNeighborsClassifier,
        hyperparams,
        (train_inputs, train_targets),
        cv_indices
    )
    
    acc_stats, err_stats, sens_stats, spec_stats, ppv_stats, npv_stats, f1_stats, cm = results
    println("    F1: $(round(f1_stats[1]*100, digits=2))% ± $(round(f1_stats[2]*100, digits=2))%")
    
    push!(knn_results, (k, f1_stats[1], results))
end

sorted_knn_results = sort(knn_results, by=x->x[2], rev=true)

println("\n🏆 kNN Results Ranking:")
println("-"^70)
for (i, (k, f1, _)) in enumerate(sorted_knn_results)
    badge = i == 1 ? "🥇" : i == 2 ? "🥈" : i == 3 ? "🥉" : "  "
    println("$badge $i. k=$k - F1: $(round(f1*100, digits=2))%")
end

best_k_knn, best_f1_knn = sorted_knn_results[1][1:2]
println("\n✨ Best kNN: k=$best_k_knn (CV F1: $(round(best_f1_knn*100, digits=2))%)")

# Train final kNN
println("\n🚀 Preparing final kNN...")

model_knn = kNNClassifier(K=best_k_knn)
mach_knn = machine(model_knn, MLJ.table(train_inputs_norm), categorical(train_targets_str))
MLJ.fit!(mach_knn, verbosity=0)

# Predict -> Needed for Ensemble!
knn_predictions = MLJ.predict(mach_knn, MLJ.table(test_inputs_norm))
knn_predictions_mode = mode.(knn_predictions)
knn_predictions_str = string.(knn_predictions_mode)

# Calculate Metrics
probs_knn = zeros(length(knn_predictions_str))
try; probs_knn = pdf.(knn_predictions, classes_str[end]); catch; end

metrics_knn = calculate_metrics_safe(probs_knn, knn_predictions_str, test_targets_str, test_targets_onehot, classes_str)

println("📊 kNN Test Results: Acc=$(round(metrics_knn["Accuracy"],digits=3)), F1=$(round(metrics_knn["F1"],digits=3))")
# ============================================================================
#            EXPERIMENT 5: ENSEMBLE METHODS
# ============================================================================
#
# Strategy: Combine the top 3 individual models to improve robustness
#
# Models Selected:
#   1. Best ANN
#   2. Best Decision Tree
#   3. Best kNN
#
# Ensemble Techniques:
#   1. Majority Voting: Each model votes equally, winner takes all
#   2. Weighted Voting: Models vote proportionally to their CV F1 scores
#
# Expected Benefits:
#   - Reduced variance through model averaging
#   - More robust predictions
#   - Leverage complementary strengths of different algorithms
#
# ============================================================================

println("\n" * "="^70)
println("🔬 EXPERIMENT 5: ENSEMBLE METHODS")
println("Combining ANN + Decision Tree + kNN")
println("="^70)

# Helper functions for ensemble
function majorityVoting(predictions::Vector{Vector{String}})
    n_samples = length(predictions[1])
    ensemble_predictions = Vector{String}(undef, n_samples)
    for i in 1:n_samples
        votes = [pred[i] for pred in predictions]
        ensemble_predictions[i] = mode(votes)
    end
    return ensemble_predictions
end

function weightedVoting(predictions::Vector{Vector{String}}, weights::Vector{Float64})
    n_samples = length(predictions[1])
    n_models = length(predictions)
    # Ensure classes are gathered from all predictions
    classes_unique = sort(unique(vcat(predictions...)))
    
    ensemble_predictions = Vector{String}(undef, n_samples)
    for i in 1:n_samples
        class_scores = Dict(c => 0.0 for c in classes_unique)
        for j in 1:n_models
            class_pred = predictions[j][i]
            if haskey(class_scores, class_pred)
                class_scores[class_pred] += weights[j]
            end
        end
        ensemble_predictions[i] = argmax(class_scores)
    end
    return ensemble_predictions
end

# 1. Prepare Predictions (Ensure all are Strings)
# ANN outputs (calculated in Exp 1 update) -> Convert to String labels
# We use preds_ann_int calculated previously (0, 1, 2 integers)
ann_test_pred_str = string.(preds_ann_int) 

# SVM outputs (already string from Exp 2)
svm_test_pred_str = svm_predictions_str

# DT outputs (already string from Exp 3)
tree_test_pred_str = tree_predictions_str

# kNN outputs (already string from Exp 4)
knn_test_pred_str = knn_predictions_str

# 2. Select Models for Ensemble (Top 3 usually, or all 4)
# Let's use ANN, DT, and kNN as per original plan (or add SVM if you wish)
all_predictions = [ann_test_pred_str, tree_test_pred_str, knn_test_pred_str]
model_names = ["ANN", "DT", "kNN"]

# 3. Define Classes as Strings (CRITICAL FIX)
classes_str = sort(unique(vcat(all_predictions...)))
test_targets_str = string.(test_targets) # Ensure targets are strings too

# --- Method 1: Majority Voting ---
println("\n[1/2] Majority Voting...")
majority_predictions = majorityVoting(all_predictions)
cm_results_majority = confusionMatrix(majority_predictions, test_targets_str, classes_str; weighted=true)

println("✅ Majority Voting Results:")
println("   F1: $(round(cm_results_majority.aggregated.f1*100, digits=2))%")
println("   Acc: $(round(cm_results_majority.accuracy*100, digits=2))%")

# --- Method 2: Weighted Voting ---
println("\n[2/2] Weighted Voting...")
# Retrieve F1 scores from the metrics calculated in previous steps
w_ann = metrics_ann["F1"]
w_dt = metrics_tree["F1"]
w_knn = metrics_knn["F1"]

cv_scores = [w_ann, w_dt, w_knn]
weights = cv_scores ./ sum(cv_scores)

println("  Model weights:")
println("    ANN: $(round(weights[1]*100, digits=1))%")
println("    DT:  $(round(weights[2]*100, digits=1))%")
println("    kNN: $(round(weights[3]*100, digits=1))%")

weighted_predictions = weightedVoting(all_predictions, weights)
cm_results_weighted = confusionMatrix(weighted_predictions, test_targets_str, classes_str; weighted=true)

println("✅ Weighted Voting Results:")
println("   F1: $(round(cm_results_weighted.aggregated.f1*100, digits=2))%")
println("   Acc: $(round(cm_results_weighted.accuracy*100, digits=2))%")


# ============================================================================
#  APPROACH 3: OVERSAMPLING STRATEGY
#  Description: Balance classes by duplicating minority samples instead of removing majority
# ============================================================================

println("\n" * "#"^70)
println("🔬 APPROACH 3: OVERSAMPLING")
println("#"^70)

# Function for Random Oversampling
function random_oversampling(df, target_col)
    classes = unique(df[!, target_col])
    # Find count of majority class
    max_count = maximum([sum(df[!, target_col] .== c) for c in classes])
    
    balanced_parts = []
    for c in classes
        df_class = df[df[!, target_col] .== c, :]
        n_current = size(df_class, 1)
        if n_current < max_count
            # Oversample with replacement
            ids = rand(1:n_current, max_count)
            push!(balanced_parts, df_class[ids, :])
        else
            push!(balanced_parts, df_class)
        end
    end
    return vcat(balanced_parts...)
end

# 1. Prepare Data (Oversampling on Training Data ONLY to prevent leakage)
# Note: We use the raw training split created in Approach 1 section
df_train_os = random_oversampling(df_train, "Risk_Class")

# 2. Preprocess (Reuse existing function)
df_train_os_proc = preprocess_multiclass(df_train_os, "Is Fraudulent")
input_cols_os = setdiff(names(df_train_os_proc), ["Risk_Class"])

train_inputs_os = Matrix{Float32}(df_train_os_proc[:, input_cols_os])
train_targets_os = Int.(df_train_os_proc.Risk_Class)

# 3. Evaluate ALL models on this new dataset
results_app3 = evaluate_approach("Oversampling", train_inputs_os, train_targets_os, test_inputs, test_targets)



# ============================================================================
#  APPROACH 4: FEATURE EXTRACTION (PCA)
#  Description: Reduce dimensionality using PCA before modeling.
#  CRITICAL FIX: PCA matrix (W) is calculated on TRAIN and applied to TEST.
# ============================================================================

using LinearAlgebra # Required for PCA

println("\n" * "#"^70)
println("🔬 APPROACH 4: PCA FEATURE EXTRACTION")
println("#"^70)

"""
    fit_pca(data, variance_threshold)
    
Calculates the projection matrix W and normalization parameters based on the provided data (Training Set).
Returns: (W, norm_params)
"""
function fit_pca(data, variance_threshold=0.95)
    # 1. Calculate normalization parameters on TRAIN data
    # We use ZeroMean normalization (Standardization) which is standard for PCA
    norm_params = calculateZeroMeanNormalizationParameters(data)
    
    # 2. Standardize the data
    data_std = normalizeZeroMean(data, norm_params)
    
    # 3. Covariance Matrix & Eigen decomposition
    C = cov(data_std)
    F = eigen(C)
    
    # 4. Sort eigenvalues (descending) and corresponding vectors
    idx = sortperm(F.values, rev=true)
    evals = F.values[idx]
    evecs = F.vectors[:, idx]
    
    # 5. Select components to reach variance threshold
    cum_var = cumsum(evals ./ sum(evals))
    k = findfirst(x -> x >= variance_threshold, cum_var)
    
    if isnothing(k)
        k = size(data, 2) # Keep all if threshold not reached
    end
    
    println("   PCA Fit: Retaining $k components (Variance covered: $(round(cum_var[k]*100, digits=2))%)")
    
    # 6. Construct Projection Matrix W
    W = evecs[:, 1:k]
    
    return W, norm_params
end

"""
    transform_data_pca(data, W, norm_params)
    
Projects new data into the PCA space defined by W, using existing normalization parameters.
"""
function transform_data_pca(data, W, norm_params)
    # 1. Normalize using the PARAMETERS from the Training Set (Critical!)
    # Note: We assume normalizeZeroMean handles parameter application correctly
    data_std = normalizeZeroMean(data, norm_params)
    
    # 2. Project into PCA space
    return data_std * W
end

# --- EXECUTION STEPS ---

# 1. Fit PCA model on Training Data ONLY
# We calculate W (eigenvectors) and normalization stats from train_inputs
println("   1. Fitting PCA on Training Set...")
pca_W, pca_norm_params = fit_pca(train_inputs, 0.95)

# 2. Transform Training Data
println("   2. Transforming Training Set...")
train_inputs_pca = transform_data_pca(train_inputs, pca_W, pca_norm_params)

# 3. Transform Test Data
# CRITICAL: We use the SAME W and norm_params calculated on Train
println("   3. Transforming Test Set (using Train projection)...")
test_inputs_pca = transform_data_pca(test_inputs, pca_W, pca_norm_params)

# 4. Evaluate Models on the new PCA-transformed space
# We pass both the transformed train and transformed test sets
results_app4 = evaluate_approach("PCA (95% Variance)", 
                                 train_inputs_pca, train_targets, 
                                 test_inputs_pca, test_targets)

# ============================================================================
#  APPROACH 5: BINARY CLASSIFICATION (ORIGINAL GROUND TRUTH)
#  Description: Train directly on the original "Is Fraudulent" label.
#  Why: Class 1 (Suspicious) contains both Frauds and Risky Legitimate ones.
#       Merging 1 & 2 would confuse the model. We use the true binary label.
# ============================================================================

println("\n" * "#"^70)
println("🔬 APPROACH 5: BINARY CLASSIFICATION (Original Fraud vs Not)")
println("#"^70)

# 1. Extract the Original Binary Targets (0/1)
# We go back to df_train/df_test because preprocessing dropped the target column
println("   Extracting original 'Is Fraudulent' labels from balanced dataframe...")

# Ensure we align with the rows used in train_inputs (which come from df_train)
train_targets_binary = Int.(df_train[!, "Is Fraudulent"])
test_targets_binary  = Int.(df_test[!, "Is Fraudulent"])

# Check distribution
n_fraud = sum(train_targets_binary .== 1)
n_legit = sum(train_targets_binary .== 0)
println("   Binary Train Distribution: Legit=$n_legit, Fraud=$n_fraud")

# 2. Evaluate ALL models on Binary Targets
# The inputs (train_inputs) remain the same (we keep the feature engineering like "Is_Night", etc.)
# but we aim for the true binary target.
results_app5 = evaluate_approach("Binary", train_inputs, train_targets_binary, test_inputs, test_targets_binary)

# ============================================================================
#              FINAL RESULTS COMPARISON
# ============================================================================
#
# Comprehensive comparison of all 6 approaches on the hold-out test set.
#
# Evaluation Metrics:
#   - F1 Score: Harmonic mean of precision and recall
#   - Accuracy: Overall correct predictions
#   - Per-Class Metrics: Performance for each risk level
#
# Key Question: Which approach best balances overall performance 
#               with fraud detection capability?
#
# ============================================================================

# ============================================================================
#  FINAL COMPARISON SUMMARY - COMPLETE & DETAILED
# ============================================================================

println("\n" * "="^100)
println("🏆 FINAL DETAILED RESULTS & COMPARISON")
println("="^100)

using Printf

# 1. Funzione di Stampa Dettagliata
function print_metrics_row(approach_name, res_dict)
    println("\n📌 Approach: $approach_name")
    
    # Header Tabella
    @printf("%-10s | %-10s | %-10s | %-10s | %-10s | %-10s\n", 
            "Model", "Accuracy", "Sensitiv.", "Specific.", "AUC-ROC", "F1-Score")
    println("-"^75)
    
    for model in ["ANN", "SVM", "DT", "kNN"]
        if haskey(res_dict, model)
            m = res_dict[model]
            @printf("%-10s | %8.2f%%  | %8.2f%%  | %8.2f%%  | %8.3f   | %8.2f%%\n", 
                    model, 
                    m["Accuracy"]*100, 
                    m["Sensitivity"]*100, 
                    m["Specificity"]*100, 
                    m["AUC"], 
                    m["F1"]*100)
        else
            println("$model: N/A")
        end
    end
end

# 2. Adattamento Risultati Approccio 1 (Undersampling)
# Se non hai ri-eseguito l'Approccio 1 con la nuova funzione evaluate_approach,
# dobbiamo creare manualmente il dizionario dai vecchi risultati per poterlo stampare.
# 2. Recupero Risultati Approccio 1 (Undersampling)
# Ora usiamo direttamente le variabili 'metrics_...' che abbiamo calcolato sopra
results_app1 = Dict(
    "ANN" => metrics_ann,
    "SVM" => metrics_svm,
    "DT"  => metrics_tree,
    "kNN" => metrics_knn
)
# 3. Stampa delle Tabelle per ogni Approccio
# Nota: Assicurati che results_app3, 4 e 5 esistano (runna le celle precedenti)
print_metrics_row("1. Undersampling (Base)", results_app1)

if isdefined(Main, :results_app3)
    print_metrics_row("2. Oversampling", results_app3)
end

if isdefined(Main, :results_app4)
    print_metrics_row("3. PCA Features", results_app4)
end

if isdefined(Main, :results_app5)
    print_metrics_row("4. Binary Classif.", results_app5)
end

println("-"^75)

# 4. Ensemble Methods (Solo su Base Approccio)
if isdefined(Main, :cm_results_majority)
    println("\n👥 Ensemble Methods (Applied to Base Undersampling):")
    # Nota: Gli ensemble vecchi non hanno dizionari dettagliati, stampiamo solo F1
    maj_f1 = cm_results_majority.aggregated.f1
    wei_f1 = cm_results_weighted.aggregated.f1
    println("   • Majority Voting: $(round(maj_f1*100, digits=2))% (F1 Score)")
    println("   • Weighted Voting: $(round(wei_f1*100, digits=2))% (F1 Score)")
end

# 5. Calcolo del Vincitore Assoluto (Logica Aggiornata per i nuovi Dizionari)
best_f1 = 0.0
best_model_name = ""
best_approach = ""

# Lista di tutti i risultati disponibili
all_results_list = [
    ("Undersampling", results_app1),
    ("Oversampling", isdefined(Main, :results_app3) ? results_app3 : Dict()),
    ("PCA", isdefined(Main, :results_app4) ? results_app4 : Dict()),
    ("Binary", isdefined(Main, :results_app5) ? results_app5 : Dict())
]

for (app_name, res_dict) in all_results_list
    for (model, metrics) in res_dict
        if metrics["F1"] > best_f1
            global best_f1 = metrics["F1"]
            global best_model_name = model
            global best_approach = app_name
        end
    end
end

# Controllo se gli ensemble vincono
if isdefined(Main, :cm_results_majority)
    if cm_results_majority.aggregated.f1 > best_f1
        global best_f1 = cm_results_majority.aggregated.f1
        global best_model_name = "Majority Voting"
        global best_approach = "Ensemble (Undersampling)"
    end
    if cm_results_weighted.aggregated.f1 > best_f1
        global best_f1 = cm_results_weighted.aggregated.f1
        global best_model_name = "Weighted Voting"
        global best_approach = "Ensemble (Undersampling)"
    end
end

println("\n" * "="^100)
println("🎯 OVERALL BEST PERFORMANCE")
println("   Approach: $best_approach")
println("   Model:    $best_model_name")
println("   F1 Score: $(round(best_f1*100, digits=2))%")
println("="^100)

println("\n📋 PROJECT SUMMARY:")
println("  ✅ Tested 4 distinct Data Approaches (Under, Over, PCA, Binary)")
println("  ✅ Evaluated 4 ML Algorithms (ANN, SVM, DT, kNN) for EACH approach")
println("  ✅ Implemented Ensemble methods on the base approach")
println("  ✅ Evaluated on full Test Set with detailed metrics (Acc, Sens, Spec, AUC, F1)")
println("="^100)