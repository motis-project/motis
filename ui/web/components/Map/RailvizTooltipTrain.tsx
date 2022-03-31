import React, { useEffect, useState } from 'react';
import moment from 'moment';
import { Delay } from '../Overlay/Delay';

export const RailvizTooltipTrain: React.FC<{'train': any}> = (props) => {

    const [train, setTrain] = useState<any>(props.train || {names: [''],
                                                            departureTime: 0,
                                                            departureStation: '',
                                                            hasDepartureDelayInfo: false,
                                                            scheduledDepartureTime: 0,
                                                            arrivalTime: 0,
                                                            arrivalStation: '',
                                                            hasArrivalDelayInfo: false,
                                                            scheduledArrivalTime: 0});

    useEffect(() => {
        if(props.train){
            setTrain(props.train);
        }
    }, [props.train]);

    return (
        <>
            <div className='transport-name'>{train.names[0]}</div>
            <div className='departure'>
                <span className='station'>{train.departureStation}</span>
                <div className={`time ${train.hasDepartureDelayInfo ? '' : 'no-delay-infos'}`}>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{moment(train.departureTime).format('HH:mm')}</span>
                    {train.hasDepartureDelayinfo ? 
                        <Delay event={{time: train.departureTime, schedule_time: train.scheduledDepartureTime, track: '', reason: 'Schedule'}}/>
                    :
                        <></>}
                </div>
            </div>
            <div className='arrival'>
            <span className='station'>{train.arrivalStation}</span>
                <div className={`time ${train.hasArrivalDelayInfo ? '' : 'no-delay-infos'}`}>{/** muss in abhaengikeit von train gesetzt werden */}
                    <span className='schedule'>{moment(train.arrivalTime).format('HH:mm')}</span>
                    {train.hasArrivalDelayInfo ?
                        <Delay event={{time: train.arrivalTime, schedule_time: train.scheduledArrivalTime, track: '', reason: 'Schedule'}}/>
                    :
                        <></>}
                </div>
            </div>
        </>
    );
}