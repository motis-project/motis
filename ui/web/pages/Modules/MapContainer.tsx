import React from "react";
import { SimTimePicker } from "./SimTimePicker";

export const MapContainer: React.FC = () => {

    const[simTimePickerSelected, setSimTimePickerSelected] = React.useState<Boolean>(false);

    return (
        <div className="map-container">
            <div id="map-background" className="mapboxgl-map">
                <div className="mapboxgl-canary" style={{visibility: "hidden"}}></div>
                <div
                    className="mapboxgl-canvas-container mapboxgl-interactive mapboxgl-touch-drag-pan mapboxgl-touch-zoom-rotate">
                    <canvas className="mapboxgl-canvas" tabIndex={0} aria-label="Map" width="1180" height="937"
                        style={{width: "1180px", height: "937px"}}></canvas></div>
                <div className="mapboxgl-control-container">
                    <div className="mapboxgl-ctrl-top-left"></div>
                    <div className="mapboxgl-ctrl-top-right"></div>
                    <div className="mapboxgl-ctrl-bottom-left">
                        <div className="mapboxgl-ctrl" style={{display: "none"}}><a className="mapboxgl-ctrl-logo" target="_blank"
                                rel="noopener nofollow" href="https://www.mapbox.com/" aria-label="Mapbox logo"></a>
                        </div>
                    </div>
                    <div className="mapboxgl-ctrl-bottom-right">
                        <div className="mapboxgl-ctrl mapboxgl-ctrl-attrib mapboxgl-attrib-empty">
                            <div className="mapboxgl-ctrl-attrib-inner"></div>
                        </div>
                    </div>
                </div>
            </div>
            <div id="map-foreground" className="mapboxgl-map">
                <div className="mapboxgl-canary" style={{visibility: "hidden"}}></div>
                <div
                    className="mapboxgl-canvas-container mapboxgl-interactive mapboxgl-touch-drag-pan mapboxgl-touch-zoom-rotate">
                    <canvas className="mapboxgl-canvas" tabIndex={0} aria-label="Map" width="1180" height="937"
                        style={{width: "1180px", height: "937px", cursor: "default"}}></canvas></div>
                <div className="mapboxgl-control-container">
                    <div className="mapboxgl-ctrl-top-left"></div>
                    <div className="mapboxgl-ctrl-top-right"></div>
                    <div className="mapboxgl-ctrl-bottom-left">
                        <div className="mapboxgl-ctrl" style={{display: "none"}}><a className="mapboxgl-ctrl-logo" target="_blank"
                                rel="noopener nofollow" href="https://www.mapbox.com/" aria-label="Mapbox logo"></a>
                        </div>
                    </div>
                    <div className="mapboxgl-ctrl-bottom-right">
                        <div className="mapboxgl-ctrl mapboxgl-ctrl-attrib">
                            <div className="mapboxgl-ctrl-attrib-inner"><a href="https://www.openstreetmap.org/">©
                                    OpenStreetMap contributors</a></div>
                        </div>
                    </div>
                </div>
            </div>
            <div className="railviz-tooltip hidden"></div>
            <div className="map-bottom-overlay">
                <div className="sim-time-overlay" onClick={() => setSimTimePickerSelected(true)} onBlur={() => setSimTimePickerSelected(false)}>
                    <div id="railviz-loading-spinner" className="">
                        <div className="spinner">
                            <div className="bounce1"></div>
                            <div className="bounce2"></div>
                            <div className="bounce3"></div>
                        </div>
                    </div>
                    <div className="permalink" title="Permalink"><a
                            href="#/railviz/49.89335526028776/8.606607315730798/11/0/0/1603118821"><i
                                className="icon">link</i></a></div>
                    <div className="sim-icon" title="Simulationsmodus aktiv"><i className="icon">warning</i></div>
                    <div className="time" id="sim-time-overlay">19.10.2020 16:47:01</div>
                </div>
                <div className="train-color-picker-overlay">
                    <div><input type="radio" id="train-color-picker-none" name="train-color-picker"/><label
                            htmlFor="train-color-picker-none">Keine Züge</label></div>
                    <div><input type="radio" id="train-color-picker-className" name="train-color-picker"/><label
                            htmlFor="train-color-picker-className">Nach Kategorie</label></div>
                    <div><input type="radio" id="train-color-picker-delay" name="train-color-picker"/><label
                            htmlFor="train-color-picker-delay">Nach Verspätung</label></div>
                </div>
            </div>
            <div className="railviz-contextmenu hidden" style={{top: "0px", left: "0px"}}>
                <div className="item">Routen von hier</div>
                <div className="item">Routen hierher</div>
            </div>
            {simTimePickerSelected && <SimTimePicker />}
        </div>
    );
};