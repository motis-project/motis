import React from 'react';

import Head from 'next/head'

import { App } from './App/App'

export default function Index() {
    return(
      <>
        <Head>
            <title>MOTIS</title>
            <script type='text/javascript' src='js/ports.js'></script>
            <script type='text/javascript' src='js/main.js'></script>
            <script type='text/javascript' src='external_lib/mapbox-gl.js'></script>
            <script type='text/javascript' src='external_lib/gl-matrix-min.js'></script>
            <script type='text/javascript' src='external_lib/deep-equal.js'></script>
            <script type='text/javascript' src='external_lib/polyline.js'></script>
            <script type='text/javascript' src='js/railviz/webgl.js'></script>
            <script type='text/javascript' src='js/railviz/trains.js'></script>
            <script type='text/javascript' src='js/railviz/model.js'></script>
            <script type='text/javascript' src='js/railviz/render.js'></script>
            <script type='text/javascript' src='js/railviz/textures.js'></script>
            <script type='text/javascript' src='js/railviz/api.js'></script>
            <script type='text/javascript' src='js/railviz/markers.js'></script>
            <script type='text/javascript' src='js/railviz/connectionmgr.js'></script>
            <script type='text/javascript' src='js/railviz/path_base.js'></script>
            <script type='text/javascript' src='js/railviz/path_connections.js'></script>
            <script type='text/javascript' src='js/railviz/path_detail.js'></script>
            <script type='text/javascript' src='js/railviz/path_extra.js'></script>
            <script type='text/javascript' src='js/railviz/style.js'></script>
            <script type='text/javascript' src='js/railviz/main.js'></script>
            <script type='text/javascript' src='js/map_style.js'></script>
            <script type='text/javascript' src='js/map.js'></script>
        </Head>
        <div id='app-container'>
            <App />
        </div>
      </>)
}
