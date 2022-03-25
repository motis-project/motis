import React from 'react';

import moment from 'moment';

import { Translations } from '../App/Localization';
import { DatePicker } from './DatePicker';
import { Station, TripId } from '../Types/Connection';
import { TripView } from './TripView';
import { Interval } from '../Types/RoutingTypes';
import { StationEvent } from '../StationSearch/StationEvent';
import { Address } from '../Types/SuggestionTypes';


function useOutsideAlerter(ref: React.MutableRefObject<any>, setSelected : React.Dispatch<React.SetStateAction<string>>) {
    React.useEffect(() => {
        /**
         * Alert if clicked on outside of element
         */
        function handleClickOutside(event) {
            if (ref.current && !ref.current.contains(event.target)) {
                setSelected('');
            }
        }

        // Bind the event listener
        document.addEventListener("mousedown", handleClickOutside);
        return () => {
            // Unbind the event listener on clean up
            document.removeEventListener("mousedown", handleClickOutside);
        };
    }, [ref]);
}


export const SubOverlay: React.FC<{ 'translation': Translations, 'subOverlayHidden': Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>>, 'detailViewHidden': Boolean, 'scheduleInfo': Interval, 'stationEventTrigger': boolean, 'setStationEventTrigger': React.Dispatch<React.SetStateAction<boolean>>, 'station': (Station | Address), 'searchDate': moment.Moment}> = (props) => {

    // Ref tracking if the time Inputfield is focused
    const timeRef = React.useRef(null);

    // Ref tracking if the searchTrain Inputfield is focused
    const trainSearchRef = React.useRef(null);

    // Current Date
    const [subOverlayDate, setSubOverlayDate] = React.useState<moment.Moment>(null);

    // subOverlayTime stores the currently displayed Time
    const [subOverlayTime, setSubOverlayTime] = React.useState<string>('');

    // timeSelected tracks if the time input is focused
    const [timeSelected, setTimeSelected] = React.useState<string>('');

    const [trainSearchSelected, setTrainSearchSelected] = React.useState<string>('');

    // On initial render searchDate will be null, waiting for the ScheduleInfoResponse. This useEffect should fire only once.
    React.useEffect(() => {
        if (props.searchDate) {
            setSubOverlayDate(props.searchDate);
            setSubOverlayTime(props.searchDate.format('HH:mm'))
        }
    }, [props.searchDate]);

    useOutsideAlerter(timeRef, setTimeSelected);
    useOutsideAlerter(trainSearchRef, setTrainSearchSelected);

    return (
        <div className={props.subOverlayHidden ? 'sub-overlay hidden' : 'sub-overlay'}>
            <div id='sub-overlay-content'>
                {(props.stationEventTrigger) ?
                    <StationEvent translation={props.translation} station={props.station} stationEventTrigger={props.stationEventTrigger} setSubOverlayHidden={props.setSubOverlayHidden} setStationEventTrigger={props.setStationEventTrigger} searchDate={props.searchDate}/> :
                    (props.trainSelected === undefined) ?
                        <div className='trip-search'>
                            <div className='header'>
                                <div id='trip-search-form'>
                                    <div className='pure-g gutters'>
                                        <div className='pure-u-1 pure-u-sm-1-2 train-nr'>
                                            <div>
                                                <div className='label'>{props.translation.search.trainNr}</div>
                                                <div className={`gb-input-group ${trainSearchSelected}`}>
                                                    <div className='gb-input-icon'>
                                                        <i className='icon'>train</i>
                                                    </div>
                                                    <input
                                                        className='gb-input' 
                                                        tabIndex={1} 
                                                        type='number'
                                                        pattern='[0-9]+' 
                                                        id='trip-search-trainnr-input'
                                                        ref={trainSearchRef} 
                                                        onFocus={() => setTrainSearchSelected('gb-input-group-selected')}/>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                    <div className='pure-g gutters'>
                                        <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                                            <div>
                                                <DatePicker translation={props.translation}
                                                    currentDate={subOverlayDate}
                                                    setCurrentDate={setSubOverlayDate}
                                                    scheduleInfo={props.scheduleInfo} />
                                            </div>
                                        </div>
                                        <div className='pure-u-1 pure-u-sm-12-24'>
                                            <div>
                                                <div className='label'>{props.translation.search.time}</div>
                                                <div className={`gb-input-group ${timeSelected}`}>
                                                <div className='gb-input-icon'><i className='icon'>schedule</i></div>
                                                <input
                                                    className='gb-input'
                                                    ref={timeRef}
                                                    tabIndex={4} 
                                                    value={subOverlayTime}
                                                    onChange={(e) => {
                                                        setSubOverlayTime(e.currentTarget.value);
                                                        if (e.currentTarget.value.split(':').length == 2) {
                                                            let [hour, minute] = e.currentTarget.value.split(':');
                                                            if (!isNaN(+hour) && !isNaN(+minute)){
                                                                let newSearchTime = moment(props.searchDate);
                                                                newSearchTime.hour(hour as unknown as number > 23 ? 23 : hour as unknown as number);
                                                                newSearchTime.minute(minute as unknown as number > 59 ? 59 : minute as unknown as number);
                                                                setSubOverlayDate(newSearchTime);
                                                                //console.log(newSearchTime)
                                                    }}}}
                                                    onKeyDown={(e) => {
                                                        if (e.key == 'Enter'){
                                                            console.log(props.searchDate)
                                                            setSubOverlayTime(props.searchDate.format('HH:mm'));
                                                        }
                                                    }}
                                                    onFocus={() => setTimeSelected('gb-input-group-selected')}/>
                                                    <div className='gb-input-widget'>
                                                    <div className='hour-buttons'>
                                                        <div><a
                                                                className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                                                onClick={() => {
                                                                    let newSearchDate = subOverlayDate.clone().subtract(1, 'h')
                                                                    setSubOverlayDate(newSearchDate); 
                                                                    setSubOverlayTime(newSearchDate.format('HH:mm'));
                                                                }}>
                                                                <i className='icon'>chevron_left</i></a></div>
                                                        <div><a
                                                                className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                                                onClick={() => {
                                                                    let newSearchDate = subOverlayDate.clone().add(1, 'h')
                                                                    setSubOverlayDate(newSearchDate);
                                                                    setSubOverlayTime(newSearchDate.format('HH:mm'));
                                                                }}>
                                                                <i className='icon'>chevron_right</i></a></div>
                                                    </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            <div className='trips'></div>
                        </div>
                        :
                        <TripView subOverlayHidden={props.subOverlayHidden} setSubOverlayHidden={props.setSubOverlayHidden} trainSelected={props.trainSelected} setTrainSelected={props.setTrainSelected} detailViewHidden={props.detailViewHidden} translation={props.translation} displayDate={props.searchDate} />
                }
            </div>
            <div className='sub-overlay-close' onClick={() => {props.setSubOverlayHidden(true); props.setStationEventTrigger(false)}}><i className='icon'>close</i></div>
        </div>
    )
}