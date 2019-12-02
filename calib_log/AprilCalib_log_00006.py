# AprilCalib log 6
# CalibRig::mode=2d
# @ Mon Nov 25 12:08:23 2019

from numpy import array
U=array([[138.103271484375, 237.2910766601563, 203.1364898681641, 169.6231384277344, 307.7236938476563, 342.6262817382813, 110.3456726074219, 202.6738128662109, 139.4064636230469, 170.1659545898438, 270.6398620605469, 304.9497985839844, 338.8450927734375, 112.9228057861328, 203.3417663574219, 171.5082397460938, 269.3690795898438, 335.2175598144531, 141.2966766357422, 302.5411376953125, 236.0414428710938, 143.6601715087891, 173.3624420166016, 331.7201538085938, 267.7226867675781, 235.791015625, 300.1942749023438, 146.5032043457031, 119.9237442016602, 204.9150085449219, 235.4992828369141, 266.9078674316406, 297.7727966308594],
       [245.0689086914063, 250.2619934082031, 248.6196746826172, 246.8094482421875, 254.1588439941406, 255.8267669677734, 274.4814758300781, 281.0841674804688, 276.9321594238281, 278.9600524902344, 285.1611022949219, 286.6600952148438, 288.3171081542969, 304.2496948242188, 311.9286193847656, 309.3963012695313, 316.0701599121094, 318.9552001953125, 306.9435424804688, 317.7534790039063, 314.4041748046875, 335.0641784667969, 338.0252685546875, 347.4963684082031, 344.8808898925781, 343.2107238769531, 346.3412170410156, 361.227294921875, 357.8395385742188, 367.3147888183594, 369.6271057128906, 371.4924926757813, 373.0177001953125]], dtype='float64');
Xw=array([[299.5044555664063, 899.5044555664063, 699.5044555664063, 499.5044555664063, 1299.504516601563, 1499.504516601563, 99.50446319580078, 699.5044555664063, 299.5044555664063, 499.5044555664063, 1099.504516601563, 1299.504516601563, 1499.504516601563, 99.50446319580078, 699.5044555664063, 499.5044555664063, 1099.504516601563, 1499.504516601563, 299.5044555664063, 1299.504516601563, 899.5044555664063, 299.5044555664063, 499.5044555664063, 1499.504516601563, 1099.504516601563, 899.5044555664063, 1299.504516601563, 299.5044555664063, 99.50446319580078, 699.5044555664063, 899.5044555664063, 1099.504516601563, 1299.504516601563],
       [99.50446319580078, 99.50446319580078, 99.50446319580078, 99.50446319580078, 99.50446319580078, 99.50446319580078, 299.5044555664063, 299.5044555664063, 299.5044555664063, 299.5044555664063, 299.5044555664063, 299.5044555664063, 299.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 499.5044555664063, 699.5044555664063, 699.5044555664063, 699.5044555664063, 699.5044555664063, 699.5044555664063, 699.5044555664063, 899.5044555664063, 899.5044555664063, 899.5044555664063, 899.5044555664063, 899.5044555664063, 899.5044555664063],
       [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype='float64');
# After LM:
K=array([[427.4057506952462, 0, 313.678024661338],
       [0, 426.0268957752566, 250.173889655714],
       [0, 0, 1]], dtype='float64');
distCoeffs=array([-0.3799963613750674,
       0.12847098498975,
       0.0009826157020280952,
       -0.009038422043721609,
       0], dtype='float64');
CovK=array([[0.06084489215731719, 0.05823886574450823, -0.01864904223710345, -0.01555017630645877, -0.0001386984602646136, 0.0002093103357707495, 7.508673021758698e-06, 1.132862401327047e-05, -8.76441036580949e-05],
       [0.05823886574453868, 0.05672035622991388, -0.01724531464793757, -0.01316559333023432, -0.0001323734735697429, 0.0001984825265754928, 6.659581350499702e-06, 1.080621852344958e-05, -8.294735579527604e-05],
       [-0.01864904223757518, -0.01724531464834958, 0.0348372320552137, 0.007543165732642395, 3.19517642206048e-05, -5.388047223484803e-05, -2.028456750211866e-06, -1.303981929423645e-05, 3.761857149965959e-05],
       [-0.01555017630571231, -0.01316559332952046, 0.007543165731650844, 0.03500294484101898, 3.257933046768359e-05, -4.635137131704434e-05, -1.157791560837153e-05, -4.545833535995139e-06, 4.561907924859964e-06],
       [-0.0001386984602645266, -0.0001323734735696014, 3.195176421958184e-05, 3.257933046908205e-05, 7.182379643998643e-07, -1.493545083823774e-06, -1.698560453992751e-08, -2.345481377954948e-08, 8.496726766126753e-07],
       [0.0002093103357708377, 0.0001984825265754868, -5.388047223349134e-05, -4.635137131926628e-05, -1.49354508382414e-06, 3.648515133455086e-06, 3.395023221213136e-08, 2.904787894406982e-08, -2.272807516182854e-06],
       [7.508673021514845e-06, 6.659581350265949e-06, -2.028456749886066e-06, -1.157791560837784e-05, -1.69856045394686e-08, 3.395023221139735e-08, 5.863463194275e-09, 1.49514344356653e-09, -1.090700757177657e-08],
       [1.132862401339731e-05, 1.080621852355626e-05, -1.303981929417411e-05, -4.545833536360935e-06, -2.345481377983312e-08, 2.904787894441377e-08, 1.495143443686404e-09, 8.332092219534234e-09, -1.621499634220907e-08],
       [-8.764410365880093e-05, -8.294735579589602e-05, 3.761857149965866e-05, 4.561907926264944e-06, 8.496726766141411e-07, -2.272807516184863e-06, -1.090700757224505e-08, -1.621499634230904e-08, 1.511160319292361e-06]], dtype='float64');
# rms=0.342937
r0=array([0.1603876241316399,
       0.1660873671931021,
       -1.951271025900015], dtype='float64');
t0=array([-624.6589478144772,
       712.350854428619,
       1979.886300360879], dtype='float64');
Covr0=array([[4.10911582268103e-07, -3.784174520990894e-08, 2.488043421908866e-08],
       [-3.784174521259623e-08, 2.958254140338086e-07, 3.572521360108556e-08],
       [2.488043421890893e-08, 3.572521360152544e-08, 1.288684590094695e-08]], dtype='float64');
Covt0=array([[0.7523059052631863, 0.1841898685869124, 0.4140376051494465],
       [0.1841898685660113, 0.7450257045526436, 0.06545974967638951],
       [0.4140376051472549, 0.06545974969454606, 1.222129979106533]], dtype='float64');
r1=array([-0.1690240455807575,
       0.07096221632878102,
       -0.03309749149040075], dtype='float64');
t1=array([-630.0433922506232,
       -576.0476114634655,
       1795.023499018323], dtype='float64');
Covr1=array([[2.063438452148818e-07, -1.545207659696021e-08, -9.152074271106836e-09],
       [-1.545207659929625e-08, 1.67935822460125e-07, 1.648126653724935e-08],
       [-9.152074271399384e-09, 1.648126653720346e-08, 5.16288846251617e-09]], dtype='float64');
Covt1=array([[0.6092826877478964, 0.1250544004593215, 0.4593988331043966],
       [0.1250544004422166, 0.5962844252832995, 0.3783644602419041],
       [0.4593988330939237, 0.378364460257951, 1.210565361634792]], dtype='float64');
r2=array([-0.1177979999794901,
       -0.05869895618458631,
       0.6871513175998177], dtype='float64');
t2=array([-125.0978172325556,
       -1132.371008799018,
       1646.209014390638], dtype='float64');
Covr2=array([[2.146568832874873e-07, 1.725393588170322e-09, -8.251997287372602e-09],
       [1.725393585346356e-09, 1.865279878230263e-07, 2.263270531726176e-08],
       [-8.251997287785693e-09, 2.263270531710332e-08, 6.482404969437987e-09]], dtype='float64');
Covt2=array([[0.5496832957155603, 0.1143868362597813, 0.3460273172569825],
       [0.1143868362446046, 0.5277287597539063, 0.4546692469292148],
       [0.3460273172433912, 0.4546692469405941, 1.178201950544243]], dtype='float64');
r3=array([-0.2806557601636891,
       -0.001434558572195994,
       0.0624818497299094], dtype='float64');
t3=array([-724.5228175177152,
       -620.2771356194301,
       1753.554903282059], dtype='float64');
Covr3=array([[1.939873317163971e-07, -1.283770034086766e-08, -7.376491470320369e-09],
       [-1.283770034378722e-08, 1.555062090661006e-07, 2.204292453294778e-08],
       [-7.376491470840457e-09, 2.204292453280966e-08, 6.654760925876661e-09]], dtype='float64');
Covt3=array([[0.5750321548649772, 0.1191011528914031, 0.4581881156080237],
       [0.1191011528751587, 0.5758694917857999, 0.397510144905801],
       [0.4581881155974524, 0.3975101449212952, 1.184322605127288]], dtype='float64');
r4=array([-0.4509164858835287,
       0.3308451630369283,
       -1.414896069586193], dtype='float64');
t4=array([-530.9684519255468,
       707.6648950145789,
       1739.485860389694], dtype='float64');
Covr4=array([[2.762667915620468e-07, -3.710072157788012e-08, 2.101225256506911e-09],
       [-3.710072158247697e-08, 1.512029147370178e-07, 3.646842400556793e-08],
       [2.101225255117113e-09, 3.646842400587275e-08, 1.35318668950349e-08]], dtype='float64');
Covt4=array([[0.5809825445027229, 0.1416597744590372, 0.3794268258606049],
       [0.1416597744426393, 0.5741823540978597, 0.06358930612706469],
       [0.3794268258596596, 0.06358930614169299, 0.8559739721950801]], dtype='float64');
r5=array([-0.7710815627478353,
       0.8854768810802022,
       -1.223335298902931], dtype='float64');
t5=array([173.6591744062165,
       407.0253085397311,
       1983.509574387828], dtype='float64');
Covr5=array([[2.577047650294086e-07, -1.679693625017322e-08, 1.762644426823313e-08],
       [-1.679693625422079e-08, 1.474021758621392e-07, 6.605877382642133e-08],
       [1.762644426592761e-08, 6.605877382711791e-08, 4.748276591537932e-08]], dtype='float64');
Covt5=array([[0.7360699876237422, 0.1408069834470407, 0.2307606332370574],
       [0.1408069834267492, 0.7321722627937858, 0.07954880481771782],
       [0.2307606332341921, 0.07954880482842211, 0.5684978612820907]], dtype='float64');
r6=array([0.27797316626484,
       -0.07060247961026407,
       0.0679685753065027], dtype='float64');
t6=array([-1321.961092111078,
       -145.2124551848488,
       2316.822653000328], dtype='float64');
Covr6=array([[3.694836705846834e-07, -3.319384702511312e-08, 2.815806581451878e-08],
       [-3.319384702876073e-08, 3.568591671827036e-07, -6.418247451360479e-08],
       [2.815806581538882e-08, -6.418247451336413e-08, 2.304824959995063e-08]], dtype='float64');
Covt6=array([[1.068251659081703, 0.2473360994402748, 0.8444637643102727],
       [0.2473360994086003, 1.186413838275435, 0.6914092881664289],
       [0.8444637642930454, 0.6914092881989973, 2.58365898738866]], dtype='float64');
