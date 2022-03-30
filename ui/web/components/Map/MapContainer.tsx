import React, { useEffect } from 'react';
import moment from 'moment';

import { DatePicker } from "../Overlay/DatePicker";
import { Translations } from "../App/Localization";
import { RailvizContextMenu } from "./RailvizContextMenu";
import { Interval } from "../Types/RoutingTypes";
import { RailvizTooltipTrain } from "./RailvizTooltipTrain";
import { RailvizTooltipStation } from "./RailvizTooltipStation";
import { SubOverlayEvent } from '../Types/EventHistory';


const getDisplay = (base: moment.Moment, increase: number, format: string) => {
    let result = base.unix();
    result = result + increase;
    return moment.unix(result).format(format + ' HH:mm:ss')
}


export const MapContainer: React.FC<{'translation': Translations, 'scheduleInfo': Interval, 'searchDate': moment.Moment, 'mapData': any, 'subOverlayContent': SubOverlayEvent[], 'setSubOverlayContent': React.Dispatch<React.SetStateAction<SubOverlayEvent[]>>}> = (props) => {
    
    // searchTime
    // SearchTime stores the currently displayed Time
    const [searchTime, setSearchTime] = React.useState<string>(moment().format('HH:mm:ss'));

    // searchTimeSelected manipulates the div 'gb-input-group' to highlight it if focused
    const [searchTimeSelected, setSearchTimeSelected] = React.useState<string>('');

    // Ref tracking if the searchTime Inputfield is focused
    const searchTimeRef = React.useRef(null);

    const [simTimePickerSelected, setSimTimePickerSelected] = React.useState<Boolean>(false);

    const [simulationDate, setSimulationDate] = React.useState<moment.Moment>(props.searchDate);

    const [systemDate, setSystemDate] = React.useState<moment.Moment>(moment());

    const [isActive, setisActive] = React.useState<boolean>(true);

    const [seconds, setSeconds] = React.useState<number>(0);

    const [stationData, setStationData] = React.useState<string>(null);

    // On initial render searchDate will be null, waiting for the ScheduleInfoResponse. This useEffect should fire only once.
    useEffect(() => {
        if (props.searchDate) {
            setSimulationDate(props.searchDate);
            setSearchTime(props.searchDate.format('HH:mm:ss'));
        }
    }, [props.searchDate]);

    const[railvizTooltipClass, setRailvizTooltipClass] = React.useState<string>('railviz-tooltip hidden');
    const[isTrainTooltip, setIsTrainTooltip] = React.useState<boolean>(null);

    useEffect(() => {
        window.portEvents.sub('mapInitFinished', function () {
            window.portEvents.pub('mapSetLocale', props.translation.search);
        });
    });

    useEffect(() => {
        if(props.mapData){
            if(props.mapData.hoveredTrain){
                /**train stuff */
                setRailvizTooltipClass('railviz-tooltip train visible');
                setIsTrainTooltip(true);
                return;
            }
            if(props.mapData.hoveredStation){
                /**station stuff */
                setRailvizTooltipClass('railviz-tooltip station visible');
                setIsTrainTooltip(false);
                return;
            }
        }
        setRailvizTooltipClass('railviz-tooltip hidden');
    }, [props.mapData]);

    useEffect(() => {
        if (simulationDate !== null) {
            let newOffset = (isActive) ? simulationDate.diff(moment()) : systemDate.diff(moment());
            window.portEvents.pub('setTimeOffset', newOffset);
        }
    }, [simulationDate, isActive]);

    useEffect(() => {
        let intervalSimulation = null;
        if (isActive) {
          intervalSimulation = setInterval(() => {
            setSeconds(seconds => seconds + 1);
          }, 1000);
        } else if (!isActive && seconds !== 0) {
          clearInterval(intervalSimulation);
        }
        let intervalReality = null;
        if (!isActive) {
            intervalReality = setInterval(() => {
              setSeconds(seconds => seconds + 1);
            }, 1000);
        } else if (isActive && seconds !== 0) {
            clearInterval(intervalReality);
        }
        return () => {clearInterval(intervalSimulation); clearInterval(intervalReality)}
    }, [isActive, seconds]);

    useEffect(() => {
        setSystemDate(moment());
        setSeconds(0);
    }, [isActive, searchTime]);

    useEffect(() =>{
        window.portEvents.sub('showStationDetails', function(data: string){
            setStationData(data);
        });
    });

    useEffect(() => {
        if ( isActive ) {
            if (simulationDate) {
                props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: {id: stationData, name: ''}, stationTime: moment.unix(simulationDate.unix() + seconds)}]);
            }
        } else {
            props.setSubOverlayContent([...props.subOverlayContent, {id: 'stationEvent', station: {id: stationData, name: ''}, stationTime: moment()}]);
        }
    }, [stationData]);

    return (
        <div className='map-container'>
            <div id='map-background' className='mapboxgl-map'>

            </div>
            <div id='map-foreground' className='mapboxgl-map'>

            </div>
            <div className={railvizTooltipClass} style={{ top: (props.mapData ? props.mapData.mouseY + 20 : 0) +'px', left: (props.mapData ? props.mapData.mouseX - 114 : 0) +'px'}}>
                {props.mapData ?
                  isTrainTooltip ? 
                    <RailvizTooltipTrain train={props.mapData.hoveredTrain}/>
                  :
                  <RailvizTooltipStation station={props.mapData.hoveredStation}/>
                : 
                  <></>}
            </div>
            <div className="map-bottom-overlay">
                <div className="sim-time-overlay" onClick={() => setSimTimePickerSelected(!simTimePickerSelected)}>
                    <div id="railviz-loading-spinner" className="">
                        <div className="spinner">
                            <div className="bounce1"></div>
                            <div className="bounce2"></div>
                            <div className="bounce3"></div>
                        </div>
                    </div>
                    <div className='permalink' title='Permalink'><a
                        href='#/railviz/49.89335526028776/8.606607315730798/11/0/0/1603118821'><i
                            className='icon'>link</i></a></div>
                    <div className='sim-icon' title='Simulationsmodus aktiv'><i className='icon'>warning</i></div>
                    <div className='time' id='sim-time-overlay'>
                        {(isActive) ? 
                            simulationDate ?
                                getDisplay(simulationDate, seconds, props.translation.dateFormat)
                                :
                                ''
                            : 
                            moment().format(props.translation.dateFormat + ' HH:mm:ss')}
                    </div>
                </div>
                <div className='train-color-picker-overlay'>
                    <div><input type='radio' id='train-color-picker-none' name='train-color-picker' onClick={() => {
                        window.portEvents.pub('mapShowTrains', false);
                    }} /><label
                        htmlFor='train-color-picker-none'>{props.translation.railViz.noTrains}</label></div>
                    <div><input type='radio' id='train-color-picker-className' name='train-color-picker' defaultChecked onClick={() => {
                        window.portEvents.pub('mapUseTrainClassColors', true);
                        window.portEvents.pub('mapShowTrains', true);
                    }} /><label
                        htmlFor='train-color-picker-className'>{props.translation.railViz.classColors}</label></div>
                    <div><input type='radio' id='train-color-picker-delay' name='train-color-picker' onClick={() => {
                        window.portEvents.pub('mapUseTrainClassColors', false);
                        window.portEvents.pub('mapShowTrains', true);
                    }} /><label
                        htmlFor='train-color-picker-delay'>{props.translation.railViz.delayColors}</label></div>
                </div>
            </div>
            <RailvizContextMenu translation={props.translation} />
            <div className={simTimePickerSelected ? 'sim-time-picker-container' : 'sim-time-picker-container hide'}>
                <div className='sim-time-picker-overlay'>
                    <div className='title'>
                        <input type='checkbox' id='sim-mode-checkbox' name='sim-mode-checkbox' defaultChecked onClick={() => { setisActive(!isActive) }} />
                        <label htmlFor='sim-mode-checkbox'>{props.translation.simTime.simMode}</label>
                    </div>
                    <div className={isActive ? 'date' : 'date disabled'}>
                        <DatePicker translation={props.translation}
                            currentDate={simulationDate}
                            setCurrentDate={setSimulationDate}
                            scheduleInfo={props.scheduleInfo} />
                    </div>
                    <div className={isActive ? 'time' : 'time disabled'}>
                        <div className='label'>{props.translation.search.time}</div>
                        <div className={`gb-input-group ${searchTimeSelected}`}>
                            <div className='gb-input-icon'><i className='icon'>schedule</i></div>
                            <input
                                className='gb-input'
                                ref={searchTimeRef}
                                tabIndex={4}
                                value={searchTime}
                                onChange={(e) => {
                                    setSearchTime(e.currentTarget.value);
                                    if (e.currentTarget.value.split(':').length == 3) {
                                        let [hour, minute, seconds] = e.currentTarget.value.split(':');
                                        if (!isNaN(+hour) && !isNaN(+minute) && !isNaN(+seconds)) {
                                            let newSearchTime = moment(simulationDate);
                                            newSearchTime.hour(hour as unknown as number > 23 ? 23 : hour as unknown as number);
                                            newSearchTime.minute(minute as unknown as number > 59 ? 59 : minute as unknown as number);
                                            newSearchTime.seconds(seconds as unknown as number > 59 ? 59 : seconds as unknown as number);
                                            setSimulationDate(newSearchTime);
                                        }
                                    }
                                }}
                                onKeyDown={(e) => {
                                    if (e.key == 'Enter') {
                                        setSearchTime(simulationDate.format('HH:mm:ss'));
                                    }
                                }}
                                onFocus={() => setSearchTimeSelected('gb-input-group-selected')} />
                            <div className='gb-input-widget'>
                                <div className='hour-buttons'>
                                    <div><a
                                        className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'
                                        onClick={() => {
                                            let newSearchDate = simulationDate.clone().subtract(1, 'h')
                                            setSimulationDate(newSearchDate);
                                            setSearchTime(newSearchDate.format('HH:mm:ss'));
                                        }}>
                                        <i className='icon'>chevron_left</i></a></div>
                                    <div><a
                                        className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'
                                        onClick={() => {
                                            let newSearchDate = simulationDate.clone().add(1, 'h');
                                            setSimulationDate(newSearchDate);
                                            setSearchTime(newSearchDate.format('HH:mm:ss'));
                                        }}>
                                        <i className='icon'>chevron_right</i></a></div>
                                </div>
                            </div>
                        </div>
                    </div>
                    <div className='close'>
                        <i className='icon' onClick={() => setSimTimePickerSelected(false)}>close</i>
                    </div>
                </div>
            </div>
        </div>
    );
};