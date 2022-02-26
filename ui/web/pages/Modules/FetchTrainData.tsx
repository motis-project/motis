import moment from 'moment';
import React, { useEffect, useState } from 'react';
import { Connection, Station, TripId } from './ConnectionTypes';

const getTrainConnection = (lineId: string, stationId: string, targetStationId: string, targetTime: number, time: number, trainNr: number) => {
    return {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            destination: { type: "Module", target: "/trip_to_connection" },
            content_type: 'TripId',
            content: { line_id: lineId, station_id: stationId, target_station_id: targetStationId, time: time, train_nr: trainNr }
        })
    };
};

export const FetchTrainData: React.FC<{ 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>> }> = (props) => {

    const [lineId, setLineId] = useState<string>(props.trainSelected.line_id);

    const [stationId, setStationId] = useState<string>(props.trainSelected.station_id);

    const [targetStationId, setTargetStationId] = useState<string>(props.trainSelected.target_station_id);

    const [targetTime, setTargetTime] = useState<number>(props.trainSelected.target_time);

    const [time, setTime] = useState<number>(props.trainSelected.time);

    const [train, setTrain] = useState<TripId>(props.trainSelected);

    const [trainConnection, setTrainConnection] = useState<Connection>();

    useEffect(() => {
        if (!props.trainSelected) {
            let requestURL = 'https://europe.motis-project.de/?elm=tripRequest';
            fetch(requestURL, getTrainConnection(lineId, stationId, targetStationId, targetTime, time, train.train_nr))
                .then(res => res.json())
                .then((res: Connection) => {
                    console.log('Trip Request successful');
                    console.log(res);
                    setTrainConnection(res);
                });
        }
    });

    return (
        <div className='connection-details trip-view'>
            <div className='connection-info'>
                <div className='header'>
                    <div className='back'><i className='icon' onClick={() => props.setSubOverlayHidden(true)}>arrow_back</i></div>
                    <div className='details'>
                        <div className='date'>22.1.2022</div>
                        <div className='connection-times'>
                            <div className='times'>
                                <div className='connection-departure'>23:37</div>
                                <div className='connection-arrival'>00:15</div>
                            </div>
                            <div className='locations'>
                                <div>Pratteln, Schlossstrasse</div>
                                <div>Basel, Dreirosenbrücke</div>
                            </div>
                        </div>
                        <div className='summary'><span className='duration'><i className='icon'>schedule</i>38min</span><span
                            className='interchanges'><i className='icon'>transfer_within_a_station</i>Keine Umstiege</span></div>
                    </div>
                    <div className='actions'></div>
                </div>
            </div>
            <div className='connection-journey' id='sub-connection-journey'>
                <div className='train-detail train-className-9'>
                    <div className='top-border'></div>
                    <div>
                        <div className='train-box train-className-9 with-tooltip' data-tooltip='Betreiber: Basler Verkehrsbetriebe\n Zugnummer: 2135'><svg className='train-icon'>
                            <use xlinkHref='#tram'></use>
                        </svg><span className='train-name'>Str 14</span></div>
                    </div>
                    <div className='first-stop'>
                        <div className='stop past'>
                            <div className='timeline train-color-border'></div>
                            <div className='time'><span className='past'>23:37</span></div>
                            <div className='delay'></div>
                            <div className='station'><span>Pratteln, Schlossstrasse</span></div>
                        </div>
                    </div>
                    <div className='direction past'>
                        <div className='timeline train-color-border'></div><i className='icon'>arrow_forward</i>Basel, Dreirosenbrücke
                    </div>
                    <div className='intermediate-stops-toggle clickable past'>
                        <div className='timeline-container'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                        </div>
                        <div className='expand-icon'><i className='icon'>expand_more</i><i className='icon'>expand_less</i></div><span>Fahrt
                            28 Stationen (38min)</span>
                    </div>
                    <div className='intermediate-stops expanded'>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:38</span></div>
                                <div className='departure'><span className='past'>23:38</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Pratteln, Bahnhofstrasse</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:39</span></div>
                                <div className='departure'><span className='past'>23:39</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Pratteln, Gempenstrasse</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:40</span></div>
                                <div className='departure'><span className='past'>23:40</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Pratteln, Kästeli</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:41</span></div>
                                <div className='departure'><span className='past'>23:41</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Pratteln, Lachmatt</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:43</span></div>
                                <div className='departure'><span className='past'>23:43</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Rothausstrasse</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:45</span></div>
                                <div className='departure'><span className='past'>23:45</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Dorf</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:46</span></div>
                                <div className='departure'><span className='past'>23:46</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Schützenstrasse</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:47</span></div>
                                <div className='departure'><span className='past'>23:47</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Zum Park</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:48</span></div>
                                <div className='departure'><span className='past'>23:48</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Käppeli</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:49</span></div>
                                <div className='departure'><span className='past'>23:49</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Muttenz, Freidorf</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:51</span></div>
                                <div className='departure'><span className='past'>23:51</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Basel, St. Jakob</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:53</span></div>
                                <div className='departure'><span className='past'>23:53</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Basel, Zeughaus</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>23:54</span></div>
                                <div className='departure'><span className='past'>23:54</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Basel, Karl Barth-Platz</span></div>
                        </div>
                        <div className='stop past'>
                            <div className='timeline train-color-border bg'></div>
                            <div className='timeline train-color-border progress' style={{ height: '100%' }}></div>
                            <div className='time'>
                                <div className='arrival'><span className='past'>00:14</span></div>
                                <div className='departure'><span className='past'>00:14</span></div>
                            </div>
                            <div className='delay'>
                                <div className='arrival'></div>
                                <div className='departure'></div>
                            </div>
                            <div className='station'><span>Basel, Brombacherstrasse</span></div>
                        </div>
                    </div>
                    <div className='last-stop'>
                        <div className='stop past'>
                            <div className='timeline train-color-border'></div>
                            <div className='time'><span className='past'>00:15</span></div>
                            <div className='delay'></div>
                            <div className='station'><span>Basel, Dreirosenbrücke</span></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )

}