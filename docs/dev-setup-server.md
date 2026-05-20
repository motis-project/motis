# Setting up a server from a development build

## Build and run MOTIS with bundled UI

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

## Run backend and UI dev server together

Run backend from your development build and UI in watch mode:

1. Start MOTIS backend:
    ```shell
    motis/build$ ./motis server
    ```

2. In a second terminal, start the UI dev server:
    ```shell
    motis/ui$ pnpm dev
    ```

3. Open the UI with an explicit backend target:
    - `http://localhost:5173/?motis=http://localhost:8080`
    - if Vite uses another port, adapt accordingly
    - for UI-only changes you can also point to a live backend, e.g. `?motis=https://api.transitous.org`
