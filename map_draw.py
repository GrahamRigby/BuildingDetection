import gmplot
from bs4 import BeautifulSoup


view_vec = [-0.8237273518849274, 0.5669861107263963]
camera_loc = (43.668058, -79.398074)
# corresponds to a bearing of 145.459
# my camera as an xz plane fov of ~64 degrees, so we add and subtract
# 32 degrees to get our fov cone

# I used http://www.geomidpoint.com/destination/ to find find some endpoints for my
# fov cone (it takes a starting point, bearing, and distance, and returns an end point)

fov_tri_v2 = (43.62464615, -79.39542195)
fov_tri_v3 = (43.65074592, -79.34317518)

def insertapikey(fname, apikey):
    def putkey(htmltxt, apikey, apistring=None):
        """put the apikey in the htmltxt and return soup"""
        if not apistring:
            apistring = "https://maps.googleapis.com/maps/api/js?key=%s&callback=initialize&libraries=visualization&sensor=true_or_false"
        soup = BeautifulSoup(htmltxt, 'html.parser')
        soup.script.decompose() #remove the existing script tag
        body = soup.body
        src = apistring % (apikey,)
        tscript = soup.new_tag("script", src=src, async="defer")
        body.insert(-1, tscript)
        return soup

    htmltxt = open(fname, 'r').read()
    soup = putkey(htmltxt, apikey)
    newtxt = soup.prettify()
    open(fname, 'w').write(newtxt)
    print('\nKey Insertion Completed!!')


if __name__ == "__main__":

    gmap = gmplot.GoogleMapPlotter(43.662249, -79.387468, 14)
    gmap.circle(43.668058, -79.398074, 50, 'r')
    lats, lngs = zip(camera_loc, fov_tri_v2, fov_tri_v3)
    gmap.polygon(lats, lngs, 'b', closed=True)
    gmap.draw("orientation_estimate.html")
    insertapikey("orientation_estimate.html", "AIzaSyCyXudXTiAyQLyAPuol__d9zvtGEqHV80c")

