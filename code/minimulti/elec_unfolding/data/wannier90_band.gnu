set style data dots
set nokey
set xrange [0: 4.13266]
set yrange [ -1.94742 : 13.60211]
set arrow from  0.69762,  -1.94742 to  0.69762,  13.60211 nohead
set arrow from  1.39523,  -1.94742 to  1.39523,  13.60211 nohead
set arrow from  1.96483,  -1.94742 to  1.96483,  13.60211 nohead
set arrow from  2.53443,  -1.94742 to  2.53443,  13.60211 nohead
set arrow from  3.23205,  -1.94742 to  3.23205,  13.60211 nohead
set xtics (" X "  0.00000," G "  0.69762," Y "  1.39523," S "  1.96483," X "  2.53443," R "  3.23205," G "  4.13266)
 plot "wannier90_band.dat"
