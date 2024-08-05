library(ggplot2)
setwd('~/data/walsh/all/clustering2/fig6/fig6d')
obs = read.csv('~/data/walsh/all/clustering2/annotations_completed_v1_v2_dist/FB080_BA17-3/v12_obs_v1_v2.csv')
obs$v1_v2_dist = obs$v1_v2_dist**0.5
types = c('EN-IT-L3_0', 'EN-IT-L3/4_3', 'EN-ET-L5/6_4', 'EN-ET-SP-2_1', 'EN-IT-L3_3', 'EN-IT-L3/4_0', 'EN-ET-L5/6_3', 'EN-ET-SP-2_3')
colors = c('#FF97EE', '#FF00FF', '#BC00BC', '#A800A8', '#ACFC64', '#00FF00', '#00C800', '#007E00')
obs1 = obs[obs$H3 %in% types,]
obs1$H3 = factor(obs1$H3, levels = types)
names(colors) = types
pdf(file = 'ridge.pdf', width=5.58, height=3.5)
print(ggplot(obs1, aes(x = v1_v2_dist, stat(count), color=H3, fill =H3 )) + 
        geom_density(alpha=0.3)+  xlab('V1-V2 Depth') + ylab('Count') + 
        coord_flip() + scale_y_reverse() + scale_fill_manual(values=colors) + scale_color_manual(values=colors)+
        theme(legend.title = element_blank()))
dev.off()

