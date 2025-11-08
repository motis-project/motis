# Setting up a server from a development build

1. Build `motis`. Refer to the respective documentation if necessary:
   - [for Linux](linux-dev-setup.md)
   - [for Windows](windows-dev-setup.md)
   - [for macOS](macos-dev-setup.md)


2. Build the UI:
    ```shell
    motis$ cd ui
    motis/ui$ pnpm install
    motis/ui$ pnpm -r build
    ```
   
3. Move the UI build into the build folder of `motis`:
    ```shell
    motis$ mv ui/build build/ui
    ```
   
4. Copy the tiles profiles to the `motis` build folder:
    ```shell
    motis$ cp -r deps/tiles/profile build/tiles-profiles
    ```
   
5. Download OpenStreetMap and timetable datasets and place them in the build folder of `motis`:
    ```shell
    motis/build$ wget https://github.com/motis-project/test-data/raw/aachen/aachen.osm.pbf
    motis/build$ wget https://opendata.avv.de/current_GTFS/AVV_GTFS_Masten_mit_SPNV.zip
    ```
   
6. Run `motis config` on the downloaded datasets to create a config file:
    ```shell
    motis/build$ ./motis config aachen.osm.pbf AVV_GTFS_Masten_mit_SPNV.zip
    ```

7. Run `motis import` and then start the server using `motis server`:
    ```shell
    motis/build$ ./motis import
    motis/build$ ./motis server
    ```

8. Open `localhost:8080` in a browser to see if everything is working.
