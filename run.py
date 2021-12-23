from core.otsu_binarization import OtsuBinarization
from core.sauvola_binarization import SauvolaBinarization
if __name__ == '__main__':
    ob = OtsuBinarization()
    ob.read_img('examples/lena.jpg')
    ob.binarize()
    ob.show_img(f'Threshold {ob.binarize_thr}')
    ob.save_img('results/otsu.jpg')

    sb = SauvolaBinarization()
    sb.read_img('examples/lena.jpg')
    sb.binarize()
    sb.show_img()
    sb.save_img('results/sauvola.jpg')