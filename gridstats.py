"""Get stats for shapes in a grid

Copied from pystar.py 20110915. pystar.py, named for someone elses older tool
doing similiar things, has too much in main() and is confounded with stream /
DEM analysis code, this will be stripped to a grid shape stats specific library
form.

REMEMBER: to get partial pixel values, draw into a
higher res. grid and resample.

"""

import optparse
import sys
import os
# import traceback

from osgeo import gdal
from osgeo import gdalconst
from osgeo import gdal_array
from osgeo import ogr
from osgeo import osr
import numpy

from math import sqrt, sin, cos, atan2, pi

import unittest
def make_parser():
    """return OptionParser for this module"""

    parser = optparse.OptionParser()

    parser.add_option("--fields", action="append", default=[],
        help="Include fields as identification in output, may be repeated "
             "and may include comma (not space) separated values. "
             "'#' can be used to output record number (starting at 1) "
             "'*' can be used to get all fields [NOT IMPLEMENTED]")

    parser.add_option("--categorical", default=False, action='store_true',
        help="Return count,class for categorical grids")

    parser.add_option("--constant", action="append", default=[],
        help="Add a constant value to the output, e.g. "
             "--constant=batch,7 to add column 'batch' to the "
             "output with a constant value of 7")

    parser.add_option("--no-header", default=False, action='store_true',
        help="Append to output, don't write header line")

    parser.add_option("--run-tests", default=False, action='store_true',
        help="Run tests, ignoring all other inputs / parameters")

    parser.add_option("--output", metavar='FILENAME',
        help="Write output to named file, over-written without warning")

    parser.add_option("--progress", type='int', default=100, metavar='N',
        help="Report progress every N shapes, 0 to suppress")

    parser.add_option("--max-records", type='int', default=0, metavar='N',
        help="Stop after processing N shapes")

    parser.add_option("--save-masks", default="", metavar='FILENAME',
        help="Specify pattern to save mask grids as geotiffs,"
        " e.g. 'mask%06d.tif'")

    parser.add_option("--save-data", default="", metavar='FILENAME',
        help="Specify pattern to save data (in shape pixels) as"
        " text, e.g. 'data%06d.txt'")

    parser.add_option("--nodata", type='float', default=None,
        help="NoData value")

    return parser

def getOptArg(args=None):
    """return opt, arg - options and arguments from OptionParser"""

    opt, arg = make_parser().parse_args(args)

    # split comma containing values in opt.fields
    for n in range(len(opt.fields)-1, -1, -1):
        opt.fields.extend([i.strip() for i in opt.fields[n].split(',')])
        del opt.fields[n]

    if not opt.fields:
        opt.fields = ['#']

    if arg:
        opt.shapes = arg[0]
        opt.grid = arg[1]
    else:
        opt.shapes = None
        opt.grid = None

    return opt, arg
def clip(val, rng):
    """return val clipped by range [0-rng]"""

    return max(min(val, rng-1), 0)
class GridClipper:
    """Clip out part of a grid, with padding, based on geometry"""


    def __init__(self, grid):

        self.img = grid
        self.rows = self.img.RasterYSize
        self.cols = self.img.RasterXSize
        self.bands = self.img.RasterCount

        (self.xorg, self.xsz,
            dummy, self.yorg, dummy, self.ysz) = self.img.GetGeoTransform()  

        self.srs = osr.SpatialReference()
        self.srs.SetUTM(15, 1)
        self.srs.SetWellKnownGeogCS('NAD83')
        self.srs_wkt = self.srs.ExportToWkt()

        self.memrefs = []
    def get_bounds(self, geom):
        """work out the cell parameters to bound geom in our grid"""

        padding = 2

        minx, maxx, miny, maxy = geom.GetEnvelope()

        # how many columns over do we start
        fcols = int((minx - self.xorg) / self.xsz)
        # add some padding
        fcols -= padding
        # how many columns over do we end
        fcole = int((maxx - self.xorg) / self.xsz)
        # add some padding
        fcole += padding

        # how many rows down do we start
        frows = int((self.yorg - maxy) / -self.ysz)
        # add some padding
        frows -= padding
        # how many rows down do we end
        frowe = int((self.yorg - miny) / -self.ysz)
        # add some padding
        frowe += padding

        # clip to boundaries of data grid
        fcols = clip(fcols, self.img.RasterXSize)
        fcole = clip(fcole, self.img.RasterXSize)
        frows = clip(frows, self.img.RasterYSize)
        frowe = clip(frowe, self.img.RasterYSize)

        # column and row counts
        fcolcnt = fcole-fcols+1
        frowcnt = frowe-frows+1

        ans = type('o', (), {})

        ans.padding = padding
        ans.xsz = self.xsz
        ans.ysz = self.ysz
        ans.minx = minx
        ans.maxx = maxx
        ans.miny = miny
        ans.maxy = maxy
        ans.fcols = fcols
        ans.fcole = fcole
        ans.frows = frows
        ans.frowe = frowe
        ans.fcolcnt = fcolcnt
        ans.frowcnt = frowcnt

        return ans
    def get_rowcol(self, bnds, x, y):
        """return row,col for x,y in a clipped raster
        given bnds from get_bounds"""

        col = int((x - self.xorg - self.xsz * bnds.fcols) / self.xsz)
        row = int((y - self.yorg - self.ysz * bnds.frows) / self.ysz)

        return row, col
    def get_clip(self, bnds):
        """get the part of our grid clipped by bnds (from get_bounds())"""
        try:
            rasterDat = gdal_array.BandReadAsArray(self.img.GetRasterBand(1),
                bnds.fcols, bnds.frows, bnds.fcolcnt, bnds.frowcnt)
        except:
            print bnds.fcols, bnds.fcole, bnds.fcolcnt, bnds.frows, \
                bnds.frowe, bnds.frowcnt
            raise

        return rasterDat

class TestExtract(unittest.TestCase):
    "set of tests"

    def chk_input(self, files, hashval):
        """check that md5 hex digest of all the files globbed from the
        list of glob expressions `files` is hashval
        """

        import hashlib, glob
        m = hashlib.md5()
        for g in files:
            for f in sorted(glob.glob(g)):
                # sorting is important for consistent results
                m.update(open(f).read())

        self.failIf(m.hexdigest() != hashval,
            "hash fail for "+','.join(files))
    def testOldResults(self):
        "get old zonal stats results"

        # this is just a "let's not break things" test which invokes
        # the main module as a subprocess with known inputs to test
        # for an expected outcome
        # 
        # this test could be broken by a change in precision or format
        # in upstream components

        import subprocess
        from hashlib import md5 

        ans = "./pystar-test-data/ans_20101015.txt"
        anshash = "551753bbffd2bf95e50565ba42ef67b7"

        self.chk_input([ans], anshash)

        shapes = "./pystar-test-data/testshapes2.shp"
        grid = "./pystar-test-data/testgrid"

        self.chk_input(["./pystar-test-data/testgrid/*"],
            "be4050353053028d4cb069e3733f92a3")

        self.chk_input(["./pystar-test-data/testshapes2.*"],
            "7604e37bdd071b15f624ed6f26ce7ae2")

        cmd = subprocess.Popen(["python", "pystar.py", shapes, grid], 
            stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out,err = cmd.communicate()

        self.failIf(md5(out).hexdigest() != anshash,
            "didn't get expected answer")
class ZonalStats(object):
    """Calculate Zonal Statistics for geometries on a grid"""

    driverMEMgdal = gdal.GetDriverByName("MEM")
    driverMEMogr = ogr.GetDriverByName('Memory')
    driverSHPogr = ogr.GetDriverByName('ESRI Shapefile')

    def __init__(self, opt=None):

        self.memrefs = []
        
        if not opt:
            self.opt, dummy = getOptArg(None)
        else:
            self.opt = opt
        self.init_from_opt()

        # register all of the GDAL drivers
        gdal.AllRegister()

    def __del__(self):

        for i in self.memrefs:
            if hasattr(i, 'destroy'):
                print 'destroying', i
                i.destroy()

        print "ZonalStats finished, now let's segfault..."
    def set_grid(self, grid):
        
        if not grid:
            return
        
        self.grid = gdal.Open(grid, gdalconst.GA_ReadOnly)
        if self.grid is None:
            print('Could not open {0}'.format(grid))
            sys.exit(1)
            
        self.memrefs.append(self.grid)
            
        self.gridfilename = grid
        self.gc = GridClipper(self.grid)
        
        self.memrefs.append(self.gc)
    def set_shapes(self, shapes):
        
        if not shapes:
            return

        self.fieldnames = [i if i != '#' else 'REC_NUM' for i in self.opt.fields]

        stats = 'cellcount', 'min', 'max', 'mean', 'std', 'sum'
        catstats = 'cellcount', 'class'

        if self.opt.categorical:
            self.out_fields = catstats
        else:
            self.out_fields = stats

        self.fieldnames.extend(self.out_fields)
        
        for i in self.opt.constant:
            self.fieldnames.append(i.split(',',1)[0])

        self.datasource = self.driverSHPogr.Open(shapes, 0)
        if self.datasource is None:
            print('Could not open {0}'.format(shapes))
            sys.exit(1)
        self.memrefs.append(self.datasource)
    def init_from_opt(self):
        
        if not self.opt.output:
            self.out = sys.stdout
        else:
            self.out = open(self.opt.output, 'a' if self.opt.no_header else 'w')

        # open the data source
        self.set_shapes(self.opt.shapes)

        # open the image
        self.set_grid(self.opt.grid)

    def run(self):
        # get the data layer
        self.layer = self.datasource.GetLayer()

        gc = self #X ZonalStats(img, opt)
        img = self.grid

        # get image size
        rows = img.RasterYSize
        cols = img.RasterXSize
        bands = img.RasterCount

        # print("{0} {1} {2}".format(rows, cols, bands))
        xorg,xsz,dummy,yorg,dummy,ysz = img.GetGeoTransform()
        # print xorg,xsz,dummy,yorg,dummy,ysz

        # yorg_orig = yorg
        # yorg = yorg_orig + rows * ysz
        # assert yorg_orig < yorg

        if not self.opt.no_header:
            self.out.write("%s\n" % ','.join(self.fieldnames))

        # loop through the features in the layer
        self.layer.ResetReading()
        feature = self.layer.GetNextFeature()
        cnt = 0

        while feature:

            cnt += 1

            if cnt == self.opt.max_records:
                break

            geom = feature.GetGeometryRef()

            output_base = [feature.GetFieldAsString(i)
                if i != '#' else str(cnt) for i in self.opt.fields]

            id_ = ':'.join([str(i) for i in output_base])

            def multi_out(func):

                results = func()

                while True:
                    try:
                        result = results.next()
                    except StopIteration:
                        return
                    except ValueError:
                        ok = False
                        result = [-9999]*len(self.out_fields)

                    output = list(output_base)
                    output.extend([("%20.10g"%i).strip() for i in result])
                    for i in self.opt.constant:
                        output.append(i.split(',',1)[-1])
                    self.out.write("%s\n" % ','.join(output))

            if not geom:
                result = [-9999]*len(self.out_fields)
                output.extend([("%20.10g"%i).strip() for i in result])
                for i in self.opt.constant:
                    output.append(i.split(',',1)[-1])
                out.write("%s\n" % ','.join(output))
            elif self.opt.categorical:
                multi_out(lambda: gc.getStatsCategorical(geom))
            else:
                multi_out(lambda: gc.getStatsContinuous(geom))

            if self.opt.save_masks:
                gdal.GetDriverByName('GTiff').CreateCopy(
                    self.opt.save_masks%cnt, gc.mask)

            if self.opt.save_data:
                of =open(self.opt.save_data%cnt, 'w') 
                of.write('\n'.join("%20.10g"%i for i in gc.data))
                of.write('\n')
                of.close()

            # dst_ds = None
            # outDs.Release()
            # outDs = None

            feature.Destroy()
            feature = self.layer.GetNextFeature()

            if self.opt.progress and (cnt % self.opt.progress == 0):
                print("Records: {0}".format(cnt))

        # close the data source
        #X self.datasource.Destroy()

        self.out.flush()

        """ 
        # dump output, R code

        interp_normals = []
        for i in range(len(normals)/2):
            interp_normals.extend(interp_points(normals[2*i:2*i+1+1],
                self.opt.normal_resolution))

        for w,n in ((coord, 'coord'), (pnts, 'pnts'), (normals, 'normals'),
                    (interp_normals, 'interp')):
            out = open(n, 'w')
            out.write('x,y\n')
            for i in w:
                out.write('%f,%f\n' % (i[0], i[1]))

        # coord=read.csv('coord'); pnts=read.csv('pnts');
        # normals=read.csv('normals'); interp=read.csv('interp')

        # plot(interp, asp=1,col='orange'); lines(coord);
        # points(coord, pch=0); points(normals, col='#008000', pch=0)

        # raise SystemExit
        """
    def getStatsContinuous(self, geom):
        """Return zonal statistics for a geometry on a continuous value grid"""

        i = self.getStats(geom)
        yield i.size, i.min(), i.max(), i.mean(), i.std(), i.sum()
    def getStatsCategorical(self, geom):
        """Return zonal statistics for a geometry on a categorical value grid"""

        not_in_i = type('nii', (), {})

        last_val = not_in_i()
        count = 0

        i = self.getStats(geom)
        i.sort()

        for j in i:
            if j != last_val:
                if count:
                    yield count, last_val 
                count = 1
                last_val = j
            else:
                count += 1

        if count:
            yield count, last_val 

    def getStats(self, geom):
        """Get stats for this geometry on our grid"""

        bnds = self.gc.get_bounds(geom)

        # create in memory grid to hold shape
        dst_ds = self.driverMEMgdal.Create('shape', 
            bnds.fcolcnt, 
            bnds.frowcnt,
            1, gdal.GDT_Byte)

        self.mask = dst_ds  # in case caller wants to save it

        trans = [
            self.gc.xorg+bnds.fcols*self.gc.xsz,
            self.gc.xsz, 
            0,
            self.gc.yorg+bnds.frows*self.gc.ysz,
            0,
            self.gc.ysz
        ]
        dst_ds.SetGeoTransform(trans)

        dst_ds.SetProjection(self.gc.srs_wkt)

        dst_ds.GetRasterBand(1).Fill(0)

        # create output shape data source
        # (20100104 - only RasterizeLayer is available in Python,
        # later we may not need the datasource / layer wrapper)
        outDs = self.driverMEMogr.CreateDataSource('mem')

        outLayer = outDs.CreateLayer('shape', geom_type=ogr.wkbPolygon,
            srs=self.gc.srs)

        feat = ogr.Feature(outLayer.GetLayerDefn())

        # defer segfault to end of execution
        self.gc.memrefs.append(feat)

        # copy shape into layer
        feat.SetGeometryDirectly(geom)

        outLayer.CreateFeature(feat)

        # rasterize
        err = gdal.RasterizeLayer(dst_ds, [1], outLayer, burn_values = [255],
            pfnTransformer = None)
        # if not pixels selected, try again more greedily
        if not err and dst_ds.GetRasterBand(1).Checksum() == 0:
            err = gdal.RasterizeLayer(dst_ds, [1], outLayer,
                burn_values = [255], options = ["ALL_TOUCHED=TRUE"],
                pfnTransformer = None)

        if err != 0:
            print('RasterizeLayer returned error', err)
            sys.exit(1)

        # numpy array of 0/not-0 pixels corresponding to shape
        raster01 = dst_ds.GetRasterBand(1).ReadAsArray()

        # extract equivalent part of data grid
        rasterDat = self.gc.get_clip(bnds)

        # get 1d array of the not-0 pixels
        data = rasterDat[raster01 > 0]

        if self.opt.nodata:
            data = data[data != self.opt.nodata]

        self.data = data  # in case caller wants it

        return data


def main():

    opt,arg = getOptArg(sys.argv[1:])

    if opt.run_tests:
        unittest.main(argv=[sys.argv[0], '--verbose'])
        return
        
    zs = ZonalStats(opt)
    zs.run()

    return

    opt.constant = ['batch,7']
    zs = ZonalStats(opt)
    zs.run()

    zs.set_grid('pystar-test-data/testgrid')
    zs.opt.constant = ['batch,42']
    zs.opt.no_header = True
    zs.run()

    #! del zs  # causes segfault

    zs2 = ZonalStats()
    zs2.opt.fields = '#', 'id'
    
    # produces a lot of output, seeing testgrid is really continuous
    # zs2.opt.categorical = True
    
    zs2.opt.constant = ['otters,several']
    zs2.set_shapes('pystar-test-data/testshapes2.shp')
    zs2.set_grid('pystar-test-data/testgrid')
    zs2.run()
if __name__ == "__main__":

    main()
