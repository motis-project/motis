import React, { useEffect } from "react";
import moment from "moment";

import { DatePicker } from "./DatePicker";
import { Translations } from "./Localization";
import { RailvizContextMenu } from "./RailvizContextMenu";

export const MapContainer: React.FC<{'translation': Translations}> = (props) => {

    const [simTimePickerSelected, setSimTimePickerSelected] = React.useState<Boolean>(false);
    
    const[currMoment, setCurrMoment] = React.useState<moment.Moment>(moment());

    useEffect(() => {
        window.portEvents.sub('mapInitFinished', function(){
            window.portEvents.pub('mapSetLocale', props.translation.search);
        });
    });

    return (
        <div className="map-container">
            <div id="map-background" className="mapboxgl-map">
                
            </div>
            <div id="map-foreground" className="mapboxgl-map">
            
            </div>
            <div className="railviz-tooltip hidden"></div>
            <div className="map-bottom-overlay">
                <div className="sim-time-overlay" onClick={() => simTimePickerSelected ? setSimTimePickerSelected(false) : setSimTimePickerSelected(true)}>
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
                    <div><input type="radio" id="train-color-picker-none" name="train-color-picker" onClick={() => {
                        window.portEvents.pub('mapShowTrains', false);
                    }}/><label
                        htmlFor="train-color-picker-none">Keine Züge</label></div>
                    <div><input type="radio" id="train-color-picker-className" name="train-color-picker" checked onClick={() => {
                        window.portEvents.pub('mapSetUseTrainClassColor', true);
                        window.portEvents.pub('mapShowTrains', true);
                    }}/><label
                        htmlFor="train-color-picker-className">Nach Kategorie</label></div>
                    <div><input type="radio" id="train-color-picker-delay" name="train-color-picker" onClick={() => {
                        window.portEvents.pub('mapSetUseTrainClassColor', false);
                        window.portEvents.pub('mapShowTrains', true);
                    }}/><label
                        htmlFor="train-color-picker-delay">Nach Verspätung</label></div>
                </div>
            </div>
            <RailvizContextMenu translation={props.translation}/>
            <div className={simTimePickerSelected ? "sim-time-picker-container" : "sim-time-picker-container hide"}>
                <div className="sim-time-picker-overlay">
                    <div className="title">
                        <input type="checkbox" id="sim-mode-checkbox" name="sim-mode-checkbox" />
                        <label htmlFor="sim-mode-checkbox">Simulationsmodus</label>
                    </div>
                    <div className="date">
                        <DatePicker translation={props.translation}
                                    currentDate={currMoment}
                                    setCurrentDate={setCurrMoment}/>
                    </div>
                    <div className="time">
                        <div className="label">Uhrzeit</div>
                        <div className="gb-input-group">
                            <div className="gb-input-icon"><i className="icon">schedule</i></div>
                            <input className="gb-input" tabIndex={21} />
                            <div className="gb-input-widget">
                                <div className="hour-buttons">
                                    <div><a
                                        className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                            className="icon">chevron_left</i></a></div>
                                    <div><a
                                        className="gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select"><i
                                            className="icon">chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className="close">
                        <i className="icon" onClick={() => setSimTimePickerSelected(false)}>close</i>
                    </div>
                </div>
            </div>
        </div>
    );
};