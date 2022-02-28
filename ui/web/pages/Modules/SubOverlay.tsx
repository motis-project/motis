import React from 'react';
import { Translations } from './Localization';
import { DatePicker } from './DatePicker';
import moment from 'moment';
import { TripId } from './ConnectionTypes';
import { FetchTrainData } from './TripView';

export const SubOverlay: React.FC<{ 'translation': Translations, 'subOverlayHidden': Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>, 'trainSelected': TripId, 'setTrainSelected': React.Dispatch<React.SetStateAction<TripId>> }> = (props) => {

    // Current Date
    const [searchDate, setSearchDate] = React.useState<moment.Moment>(moment());

    return (
        <div className={props.subOverlayHidden ? 'sub-overlay hidden' : 'sub-overlay'}>
            <div id='sub-overlay-content'>
                {(props.trainSelected === undefined) ?
                    <div className='trip-search'>
                        <div className='header'>
                            <div id='trip-search-form'>
                                <div className='pure-g gutters'>
                                    <div className='pure-u-1 pure-u-sm-1-2 train-nr'>
                                        <div>
                                            <div className='label'>Zugnummer</div>
                                            <div className='gb-input-group'>
                                                <div className='gb-input-icon'>
                                                    <i className='icon'>train</i>
                                                </div>
                                                <input
                                                    className='gb-input' tabIndex={1} type='number'
                                                    pattern='[0-9]+' id='trip-search-trainnr-input' />
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                <div className='pure-g gutters'>
                                    <div className='pure-u-1 pure-u-sm-12-24 to-location'>
                                        <div>
                                            <DatePicker translation={props.translation}
                                                currentDate={searchDate}
                                                setCurrentDate={setSearchDate} />
                                        </div>
                                    </div>
                                    <div className='pure-u-1 pure-u-sm-12-24'>
                                        <div>
                                            <div className='label'>Uhrzeit</div>
                                            <div className='gb-input-group'>
                                                <div className='gb-input-icon'>
                                                    <i className='icon'>schedule</i>
                                                </div>
                                                <input className='gb-input' tabIndex={4} />
                                                <div className='gb-input-widget'>
                                                    <div className='hour-buttons'>
                                                        <div><a
                                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                                className='icon'>chevron_left</i></a></div>
                                                        <div><a
                                                            className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select'><i
                                                                className='icon'>chevron_right</i></a></div>
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
                    <FetchTrainData subOverlayHidden={props.subOverlayHidden} setSubOverlayHidden={props.setSubOverlayHidden} trainSelected={props.trainSelected} setTrainSelected={props.setTrainSelected} />}
            </div>
            <div className='sub-overlay-close' onClick={() => props.setSubOverlayHidden(true)}><i className='icon'>close</i></div>
        </div>
    )
}