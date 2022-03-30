import moment from 'moment';
import React from 'react';
import { Translations } from '../App/Localization';
import { useOutsideAlerter } from '../App/OutsideAlerter';
import { Station } from '../Types/Connection';
import { RailVizTripGuessResponse, Trip } from '../Types/RailvizStationEvent';
import { Interval } from '../Types/RoutingTypes';
import { DatePicker } from './DatePicker';


const getRoutingOptions = (time: moment.Moment, train_num: string) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: 'Module', target: '/railviz/get_trip_guesses' },
            content_type: 'RailVizTripGuessRequest',
            content: { guess_count: 20, time: time, train_num: train_num }
        })
    }
}


export const TripSearchHeader: React.FC<{ 'translation': Translations, 'scheduleInfo': Interval, 'searchDate': moment.Moment, 'trainNumber': string, 'setTrainNumber': React.Dispatch<React.SetStateAction<string>>, 'setTrips': React.Dispatch<React.SetStateAction<{first_station: Station, trip_info: Trip}[]>> }> = (props) => {

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

    // trainSearchSelect tracks if the trainnumber input is focused
    const [trainSearchSelected, setTrainSearchSelected] = React.useState<string>('gb-input-group-selected');
    
    // On initial render searchDate will be null, waiting for the ScheduleInfoResponse. This useEffect should fire only once.
    React.useEffect(() => {
        if (props.searchDate) {
            setSubOverlayDate(props.searchDate);
            setSubOverlayTime(props.searchDate.format('HH:mm'))
        }
    }, [props.searchDate]);

    React.useEffect(() => {
        if (subOverlayDate && props.trainNumber !== ''){
            let requestURL = 'https://europe.motis-project.de/?elm=TripSearch';

            fetch(requestURL, getRoutingOptions(subOverlayDate, props.trainNumber))
                .then(res => res.json())
                .then((res: RailVizTripGuessResponse) => {
                    props.setTrips(res.content.trips);
                    })
                .catch(_error => {})
        }

    }, [subOverlayDate, subOverlayTime, props.trainNumber]);

    useOutsideAlerter(timeRef, setTimeSelected);
    useOutsideAlerter(trainSearchRef, setTrainSearchSelected);

    return (
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
                                    type='text'
                                    id='trip-search-trainnr-input'
                                    value={props.trainNumber}
                                    ref={trainSearchRef}
                                    autoFocus
                                    onChange={(e) => {
                                        props.setTrainNumber(e.currentTarget.value);
                                    }} 
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
                                            let newSearchTime = moment(subOverlayDate);
                                            newSearchTime.hour(hour as unknown as number > 23 ? 23 : hour as unknown as number);
                                            newSearchTime.minute(minute as unknown as number > 59 ? 59 : minute as unknown as number);
                                            setSubOverlayDate(newSearchTime);
                                }}}}
                                onKeyDown={(e) => {
                                    if (e.key == 'Enter'){
                                        setSubOverlayTime(subOverlayDate.format('HH:mm'));
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
    )
}