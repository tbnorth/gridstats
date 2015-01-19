
.. |gs| replace:: ``gridstats.py``

|gs| is a Python script using the `OSGeo` `GDAL` grid processing
libraries and `NumPy` extensions to analyze grids in the same way
as `Zonal Stats` and `Tabulate Areas`.  It is, thanks to 
`GDAL <http://www.gdal.org/>`_, much faster than you would expect,
and handles overlapping polygons.

On Windows, |gs| runs in the `OSGeo4W <http://trac.osgeo.org/osgeo4w/>`_
environment, so install that.  Then run the `OSGeo4W` shell from
the Start → Programs menu. On Linux etc. the required libraries are
probably available through the package manager, e.g.::

  sudo apt-get install python-gdal

If you run::

    python gridstats.py --help

you should see::

    Usage: gridstats.py [options] <shapefile> <grid>

    Options:
      -h, --help            show this help message and exit
      --fields=FIELDS       Include fields as identification in output, may be
                            repeated and may include comma (not space) separated
                            values. '#' can be used to output record number
                            (starting at 1) '*' can be used to get all fields [NOT
                            IMPLEMENTED]
      --categorical         Return count,class for categorical grids
      --constant=CONSTANT   Add a constant value to the output, e.g.
                            --constant=batch,7 to add column 'batch' to the output
                            with a constant value of 7
      --no-header           Append to output, don't write header line
      --run-tests           Run tests, ignoring all other inputs / parameters
      --output=FILENAME     Write output to named file, over-written without
                            warning
      --progress=N          Report progress every N shapes, 0 to suppress
      --max-records=N       Stop after processing N shapes
      --save-masks=FILENAME
                            Specify pattern to save mask grids as geotiffs, e.g.
                            'mask%06d.tif'
      --save-data=FILENAME  Specify pattern to save data (in shape pixels) as
                            text, e.g. 'data%06d.txt'
      --nodata=NODATA       NoData value

but generally you want something like::

    python gridstats.py --fields=site --output=results.csv path/to/foo.shp path/to/grid

The ``--nodata=<value>`` flag may be useful for continuous variables
(the default mode).  ``--categorical`` mode switches to "Tabulate Areas"
mode.
