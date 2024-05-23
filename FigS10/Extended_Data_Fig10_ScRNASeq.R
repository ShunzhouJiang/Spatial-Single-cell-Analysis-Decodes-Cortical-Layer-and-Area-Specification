#Xuyu_Nature_Seurat_scRNASeq.R
#Author : Monica D Manam
#         Walsh Lab-BCH
#         2024
#Package Credits : Seurat - Satija Lab

#Import Libraries
library(Seurat)
require(data.table)
library(SeuratData)
library(patchwork)
library(ggplot2)
library(RColorBrewer)
library(dplyr)
set.seed(1234)


#Fig_Ext_10-f,h
######### SECTION 1 :Reading the spaceranger output of the scRNASeq experiment#

# There are three I,J,3A datasets. Showing only "J" PBMC dataset here.
pbmc.data <- Read10X(data.dir = "filtered_feature_bc_matrix_FB080_J_0918/")
# Initialize the Seurat object with the raw (non-normalized data).
pbmc <- CreateSeuratObject(counts = pbmc.data, project = "pbmc3k", 
                           min.cells = 3, min.features = 200)

#calculate mitochondrial QC metrics with the PercentageFeatureSet() function, 
#which calculates the percentage of counts originating from a set of features
#use the set of all genes starting with MT- as a set of mitochondrial genes
pbmc[["percent.mt"]] <- PercentageFeatureSet(pbmc, pattern = "^MT-")

# Visualize QC metrics as a violin plot
VlnPlot(pbmc, features = c("nFeature_RNA", "nCount_RNA", "percent.mt"), ncol = 3)

# FeatureScatter is typically used to visualize feature-feature relationships, 
#but can be used
# for anything calculated by the object, i.e. columns in object metadata, 
#PC scores etc.

FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "percent.mt")
FeatureScatter(pbmc, feature1 = "nCount_RNA", feature2 = "nFeature_RNA")


#filter cells that have unique feature counts over 2,500 or less than 200
#filter cells that have >5% mitochondrial counts
pbmc <- subset(pbmc, subset = nFeature_RNA > 200 & nFeature_RNA < 10000 &
                 percent.mt < 5)
#Normalizing the data
pbmc <- NormalizeData(pbmc, normalization.method = "LogNormalize", 
                      scale.factor = 10000)
#Identification of highly variable features (feature selection)
pbmc <- FindVariableFeatures(pbmc, selection.method = "vst", nfeatures = 2000)
#Scaling the data
all.genes <- rownames(pbmc)
pbmc <- ScaleData(pbmc, features = all.genes)

pbmc <- RunPCA(pbmc, features = VariableFeatures(object = pbmc))
VizDimLoadings(pbmc, dims = 1:2, reduction = "pca")
DimPlot(pbmc, reduction = "pca")
DimHeatmap(pbmc, dims = 1, cells = 500, balanced = TRUE)

#Cluster the cells
#seurat tutorial : 0.4-1.2 typically returns good results for single-cell 
#datasets of around 3K cells.
pbmc <- FindNeighbors(pbmc, dims = 1:50)
pbmc <- FindClusters(pbmc, resolution = 0.5)

#Run non-linear dimensional reduction (UMAP/tSNE)
pbmc <- RunUMAP(pbmc, dims = 1:50)

# find markers for every cluster compared to all remaining cells,
#report only the positive ones
brain.markers <- FindAllMarkers(pbmc, only.pos = TRUE, min.pct = 0.25, 
                                logfc.threshold = 0.25)


######### SECTION 2 : Integration ######

#Using all the three .rds files generated in section 1 
I <- readRDS("FB080_I_0918.rds")
J <- readRDS("FB080_J_0918.rds")
A <- readRDS("FB121_3A_0918.rds")

I$orig.ident = "FB080_I"
J$orig.ident = "FB080_J"
A$orig.ident = "FB121_3A"

# select features that are repeatedly variable across datasets for integration
features <- SelectIntegrationFeatures(object.list = c(I,J,A))

anchors <- FindIntegrationAnchors(object.list = c(I,J,A), 
                                  anchor.features = features)

# this command creates an 'integrated' data assay
combined <- IntegrateData(anchorset = anchors)

# specify that we will perform downstream analysis on the 
#corrected data note that the
# original unmodified data still resides in the 'RNA' assay
DefaultAssay(combined) <- "integrated"

# Run the standard workflow for visualization and clustering
combined <- ScaleData(combined, verbose = FALSE)
combined <- RunPCA(combined, npcs = 30, verbose = FALSE)
combined <- RunUMAP(combined, reduction = "pca", dims = 1:30)
combined <- FindNeighbors(combined, reduction = "pca", dims = 1:30)
#Find more clusters
combined <- FindClusters(combined, resolution = 1.0)

DefaultAssay(combined) <- "RNA"
combined.markers <- FindAllMarkers(combined, only.pos = TRUE, 
                                   min.pct = 0.25, logfc.threshold = 0.25)

#Fig_Ext_10_i
immune.combined <- RenameIdents(combined, `0` = "EN-Mig-1", `1` = "EN-L2", 
                                `2` = "EN-Mig-2", `3` = "EN-IT-UL-1", 
                                `4` = "EN-IT-UL-2", `5` = "EN-ET-1", 
                                `6` = "EN-IT-L4-V1", `7` = "EN-IT-UL-3", 
                                `8` = "EN-IT-DL-1", `9` = "IN-CGE-1", 
                                `10` = "IPC", `11` = "EN-ET-2", `12` = "RG", 
                                `13` = "EN-IT-UL-4", `14` = "EN-IT-DL-2", 
                                `15` = "EN-ET-3", `16` = "IN-MGE-1", 
                                `17` = "IN-MGE-2", `18` = "unknown", 
                                `19` = "EN-ET-4", `20` = "Dividing progenitor", 
                                `21` = "OPC", `22` = "IN-CGE-2", `23` = "vRG", 
                                `24` = "EN-ET-5", `25` = "EN-ET-6", 
                                `26` = "EN-ET-7", `27` = "EN-IT-DL-3", 
                                `28` = "IN-MGE-3", `29` = "EN-IT-UL-5", 
                                `30` = "INP-CGE", `31` = "Microglia", 
                                `32` = "EN-Mig-3", `33` = "EC-1", 
                                `34` = "EC-2")

DimPlot(immune.combined, label = TRUE)


###### Fig_Extended10-i : Contribution plot ###########################

# Store cluster identities in object@meta.data$my.clusters
immune.combined[["my.clusters"]] <- Idents(immune.combined)

# Get number of cells per cluster and per sample of origin
cluster_table <- table(immune.combined@meta.data$my.clusters, 
                       immune.combined@meta.data$orig.ident)

# Convert the table to a data frame
cluster_df <- as.data.frame.matrix(cluster_table)

# Convert row names to a new column 'Cell_Type'
cluster_df$Cell_Type <- rownames(cluster_df)

# Reshape the data to long format
library(tidyr)
cluster_df_long <- gather(cluster_df, key = "Sample", value = "Contribution", -Cell_Type)

# Calculate total contribution per sample
totals_per_sample <- cluster_df_long %>%
  group_by(Sample) %>%
  summarise(Total_Contribution = sum(Contribution))

# Merge total contributions with the original data
cluster_df_long <- merge(cluster_df_long, totals_per_sample, by = "Sample")

# Reorder samples based on total contribution (descending order)
ordered_samples <- c("FB121_3A","FB080_I","FB080_J")
custom_colors <- c("#e31a1c", "#a6cee3", "#1f78b4")
# Convert Sample to factor with ordered levels
cluster_df_long$Sample <- factor(cluster_df_long$Sample, 
                                 levels = ordered_samples)

# Create a ggplot stacked bar plot with proportions and without labels
ggplot(cluster_df_long, aes(x = Cell_Type, y = Contribution, 
                            fill = Sample)) +
  geom_bar(position = "fill", stat = "identity") +
  labs(title = "Proportion of Cell Types in Each Sample",
       x = "Cell Type", y = "Proportion") +
  scale_fill_manual(name = "Sample", 
                    values = setNames(custom_colors, 
                                      ordered_samples))
ggsave("Fig10_Extended-i.pdf", width = 30, height = 6)

#4*4 FeaturePlots
#feat_I-J-A_part1
FeaturePlot(immune.combined, features = c("PDZRN3", "DLX6", "DLX6-AS1", "ADARB2", 
                                          "ERBB4", "NRXN3", "DLX2", "ZNF536",
                                          "PRKCA", "THRB","TSHZ1","PBX3", 
                                          "MEIS2", "CALB2", "CDCA7L","SYNPR" ),
            min.cutoff = "q9",
            order = TRUE,pt.size=0.01)

#feat_I-J-A_part2
FeaturePlot(immune.combined, features = c("KCNIP4","CASZ1", "FOXP4", "SP8", 
                                          "CUX2","CCBE1","MDGA1","CDH13",
                                          "TRPC6","CHRM2","RORB","WHRN",
                                          "NELL1","ETV6","IL1RAPL2","SP8"),
            min.cutoff = "q9",
            order = TRUE,pt.size=0.01)

#feat_I-J-A_part3
FeaturePlot(immune.combined, features = c("QRFPR","FN1","THSD7B","CDH18",
                                          "GALNTL6","BRINP3","HTR4","PPARG",
                                          "PLD5","CCN2","LUZP2","LRP1B",
                                          "LRRTM4","HDAC9","FBXL7","DTNA" ),
            min.cutoff = "q9",
            order = TRUE,pt.size=0.01)

#feat_I-J-A_part4
FeaturePlot(immune.combined, features = c("SYNDIG1","SDK1","LMO3","TRIQK",
                                          "TAFA2","KCNQ5","TOX","UNC13C",
                                          "CNTNAP2","ETV1","DEPTOR","TSHZ2",
                                          "NPY","ABI3BP","PDZRN4" ),
            min.cutoff = "q9",
            order = TRUE,pt.size=0.01)

#feat_I-J-A_part5
FeaturePlot(combined, features = c("TBR1","TLE4","NR4A1","TOX","B3GALT2",
                                   "SEMA3C","CRYM","RORB","CUX2",
                                   "CBLN2","NPY"), 
            min.cutoff = "q9",order = TRUE,pt.size=0.01)
