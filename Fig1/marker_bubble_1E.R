library(anndata)
library(stringr)
library(Matrix)
library(MASS)
library(reticulate)
library(Seurat)
library(dplyr)
library(splitstackshape)
library(ggplot2)
library(reshape2)
library(cowplot)
library(readxl)

align_legend <- function(p, hjust = 0.5)
{
  # extract legend
  g <- cowplot::plot_to_gtable(p)
  grobs <- g$grobs
  legend_index <- which(sapply(grobs, function(x) x$name) == "guide-box")
  legend <- grobs[[legend_index]]
  
  # extract guides table
  guides_index <- which(sapply(legend$grobs, function(x) x$name) == "layout")
  
  # there can be multiple guides within one legend box  
  for (gi in guides_index) {
    guides <- legend$grobs[[gi]]
    
    # add extra column for spacing
    # guides$width[5] is the extra spacing from the end of the legend text
    # to the end of the legend title. If we instead distribute it by `hjust:(1-hjust)` on
    # both sides, we get an aligned legend
    spacing <- guides$width[5]
    guides <- gtable::gtable_add_cols(guides, hjust*spacing, 1)
    guides$widths[6] <- (1-hjust)*spacing
    title_index <- guides$layout$name == "title"
    guides$layout$l[title_index] <- 2
    
    # reconstruct guides and write back
    legend$grobs[[gi]] <- guides
  }
  
  # reconstruct legend and write back
  g$grobs[[legend_index]] <- legend
  # group_name <- group
  # pdf(paste0(paste0(path, group_name, sep = "/"), ".pdf",sep = ""))
  g
}

setwd("~/Desktop/RA/Xuyu_Project/marker_bubble")
marker_total <- readRDS("h2_marker.rds")

marker_num <- 1
marker_total <- marker_total[1:marker_num, ]

h2_order <- read_excel("smartseq_singler_h3.xlsx")
h2_order <- h2_order$H3_annotation

gene_tot <- c()
for (i in 1:length(h2_order)) {
  h2_i <- gsub("-", ".", h2_order[i])
  h2_i <- gsub("/", ".", h2_i)
  marker_i <- marker_total[, h2_i]
  gene_tot <- c(gene_tot, marker_i)
}
gene_tot <- unique(gene_tot)
gene_tot <- gsub("-", ".", gene_tot)


count_avg <- read.csv("result/h2_nz.csv", row.names = 1)
zs_avg <- read.csv("result/h2_zs.csv", row.names = 1)


count_select <- count_avg[, gene_tot]
count_select <- count_select * 100
zs_select <- zs_avg[, gene_tot]
count_select <- count_select[h2_order, ]
zs_select <- zs_select[h2_order, ]


count_select$layer <- rownames(count_select)
zs_select$layer <- rownames(zs_select)

nonzero_sub_melt <- melt(count_select, id = c("layer"))
expr_sub_melt <- melt(zs_select, id = c("layer"))
nonzero_sub_melt$layer <- factor(nonzero_sub_melt$layer, levels = h2_order)
expr_sub_melt$layer <- factor(expr_sub_melt$layer, levels = h2_order)
color_map <- expr_sub_melt$value
mid <- mean(color_map)

# col_min <- floor(min(zs_select[,1:(ncol(zs_select)-1)]))
col_min <- min(zs_select[,1:(ncol(zs_select)-1)]) - 0.5
# col_max <- ceiling(max(zs_select[,1:(ncol(zs_select)-1)]))
col_max <- max(zs_select[,1:(ncol(zs_select)-1)]) + 0.5

x = ggplot(nonzero_sub_melt, aes(x = layer, y = variable, size = value, color = color_map)) + 
  # geom_point(aes(size = value, fill = layer), alpha = 1, shape = 21) + 
  geom_point(aes(size = value)) + 
  scale_size_continuous(limits = c(-0.000001, 100), range = c(1,10), breaks = c(20, 40, 60, 80)) +
  labs( x= "", y = "", size = "Percentage of expressed cells (%)", fill = "", color = "Mean expression")  +
  theme(legend.title.align = 0.5,
        legend.text.align = 0.5,
        legend.key=element_blank(),
        axis.text.x = element_text(colour = "black", size = 10, face = "bold", angle = 90), 
        axis.text.y = element_text(colour = "black", face = "bold", size = 10), 
        legend.text = element_text(size = 8, face ="bold", colour ="black"), 
        legend.title = element_text(size = 8, face = "bold"), 
        panel.background = element_blank(),  #legend.spacing.x = unit(0.5, 'cm'),
        legend.position = "right",
        legend.box.just = "top",
        legend.direction = "vertical",
        legend.box="vertical",
        legend.justification="center"
  ) +  
  # theme_minimal() +
  # theme(legend.title.align = 0)+
  
  # scale_fill_manual(values = color_map, guide = FALSE) + 
  scale_y_discrete(limits = rev(levels(nonzero_sub_melt$variable))) +
  scale_color_gradient(low="yellow", high="blue", space ="Lab", limits = c(col_min, col_max), position = "bottom") +
  # scale_fill_viridis_c(guide = FALSE, limits = c(-0.5,1.5)) + 
  guides(colour = guide_colourbar(direction = "vertical", barheight = unit(4, "cm"), title.vjust = 4))

if (marker_num == 1) {
  height <- 26
} else if (marker_num == 3) {
  height <- 36
} else if (marker_num == 5) {
  height <- 42
}

ggdraw(align_legend(x))
ggsave(paste0(paste("h3_marker", marker_num, sep = "_"), ".pdf"),
       width = 38, height = height, limitsize = TRUE)
