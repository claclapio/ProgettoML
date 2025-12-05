# ============================================================================
#  VISUALIZATION & STATISTICS UTILITIES
# ============================================================================

function plot_loss_curves(train_loss, val_loss; title="Model Loss")
    p = plot(train_loss, label="Training Loss", xlabel="Epochs", ylabel="Loss", 
             title=title, lw=2, color=:blue)
    if !isempty(val_loss)
        plot!(p, val_loss, label="Validation Loss", lw=2, color=:orange)
    end
    display(p)
    # savefig("loss_curve_$(title).png") # Scommenta per salvare
end

function plot_confusion_matrix(cm, classes; title="Confusion Matrix")
    # Normalizza per heatmap (opzionale, o usa valori assoluti)
    # Qui usiamo i conteggi assoluti ma annotiamo il grafico
    heatmap(cm, 
            xticks=(1:length(classes), classes), 
            yticks=(1:length(classes), classes),
            yflip=true, 
            color=:Blues, 
            title=title,
            aspect_ratio=:equal)
            
    # Annotazioni testuali dei numeri
    nrow, ncol = size(cm)
    for i in 1:nrow, j in 1:ncol
        annotate!(j, i, text(string(cm[i,j]), 8, :black, :center))
    end
    display(current())
    # savefig("cm_$(title).png")
end

function plot_model_comparison(model_names, f1_scores_list)
    # f1_scores_list deve essere un vettore di vettori (es. [ann_f1s, svm_f1s, ...])
    # StatsPlots boxplot accetta (x, y) dove x sono le categorie
    
    data = []
    labels = []
    
    for (name, scores) in zip(model_names, f1_scores_list)
        append!(data, scores)
        append!(labels, fill(name, length(scores)))
    end
    
    p = boxplot(labels, data, title="Model Comparison (CV F1 Score)", 
                ylabel="F1 Score", legend=false, outliers=true)
    display(p)
    # savefig("model_comparison_boxplot.png")
end