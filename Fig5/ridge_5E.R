library(ggplot2)
setwd('~/data/walsh/all/clustering2/fig6/fig6e')
obs = read.csv('fb080_ba17-3_all_scshc_obs_cp.csv')
obs$cp_dist = obs$cp_dist**0.5
types = c('EN-IT-L3_0', 'EN-IT-L3/4_3', 'EN-ET-L5/6_4', 'EN-ET-SP-2_1', 'EN-IT-L3_3', 'EN-IT-L3/4_0', 'EN-ET-L5/6_3', 'EN-ET-SP-2_3')
obs1 = obs[obs$H3 %in% types,]
obs1$H3 = factor(obs1$H3, levels = types)
colors = c('#FF97EE', '#FF00FF', '#BC00BC', '#A800A8', '#ACFC64', '#00FF00', '#00C800', '#007E00')
names(colors) = types
pdf(file = 'ridge.pdf', width=5.58, height=3.5)
print(ggplot(obs1, aes(x = cp_dist, stat(count), color=H3, fill =H3 )) + 
        geom_density(alpha=0.3)+  xlab('Cortical Depth') + ylab('Count') + 
        coord_flip() + scale_y_reverse() + scale_fill_manual(values=colors) + scale_color_manual(values=colors)+
        geom_vline(xintercept=c(0.7354541900310434, 0.5103376408816647, 0.4110935664713694), linetype="dashed", color = "#1f77b4")+
        theme(legend.title = element_blank()))
dev.off()

