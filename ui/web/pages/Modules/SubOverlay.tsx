import React, { useEffect } from 'react';
import DatePicker from 'react-datepicker';

import { Station, Position } from './ConnectionTypes';


const displayToday = () => {
    let today = new Date();
    let dd = String(today.getDate()).padStart(2, '0');
    let mm = String(today.getMonth() + 1).padStart(2, '0'); //January is 0!
    let yyyy = today.getFullYear();
    return dd + '.' + mm + '.' + yyyy
}


const days = ['SO', 'MO', 'DI', 'MI', 'DO', 'FR', 'SA']
const months = ['Januar', 'Februar', 'März', 'April', 'Mai', 'Juni', 'Juli', 'August', 'September', 'Oktober', 'November', 'Dezember']


const locale = {
  localize: {
    day: n => days[n],
    month: n => months[n]
  },
  formatLong: {
    date: () => 'mm/dd/yyyy'
  }
}


function addDays(date: Date, days: number): Date {
    let res = new Date(date);
    res.setDate(res.getDate() + days);
    return res;
}


export const SubOverlay: React.FC<{'subOverlayHidden' : Boolean, 'setSubOverlayHidden': React.Dispatch<React.SetStateAction<Boolean>>}> = (props) => {

    const[datePickerSelected, setDatePickerSelected] = React.useState<Boolean>(false);

    const[currentDate, setCurrentDate] = React.useState<Date>(new Date());

    return (
        <div className={props.subOverlayHidden ? 'sub-overlay hidden' : 'sub-overlay'}>
            <div id='sub-overlay-content'>
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
                                        <div>
                                            <div className='label'>Datum</div>
                                            <div className='gb-input-group'>
                                                <div className='gb-input-icon'>
                                                    <i className='icon'>event</i></div>
                                                <DatePicker id='datepicker-suboverlay'
                                                            className='gb-input' 
                                                            calendarClassName='calendardays' 
                                                            popperClassName='popper calendar'
                                                            locale={locale}
                                                            showPopperArrow={false}
                                                            renderCustomHeader={({
                                                                date,
                                                                decreaseMonth,
                                                                increaseMonth,
                                                            }) => (
                                                                <div
                                                                    style={{
                                                                        display: "flex",
                                                                        justifyContent: "center",
                                                                    }}
                                                                    >
                                                                    <div className='month'>
                                                                        <i className='icon' onClick={decreaseMonth}>chevron_left</i>
                                                                        <span className='month-name'>{months[date.getMonth()] + ' ' + date.getFullYear()}</span>
                                                                        <i className='icon' onClick={increaseMonth}>chevron_right</i>
                                                                    </div>
                                                                </div>
                                                            )}
                                                            calendarStartDay={1}
                                                            selected={currentDate}
                                                            dateFormat='dd.MM.yyyy'
                                                            onChange={(date: Date) => setCurrentDate(date)}
                                                            />
                                                {/*<input className='gb-input' tabIndex={3} defaultValue={displayToday()} onBlur={() => setDatePickerSelected(false)} onFocus={() => setDatePickerSelected(true)}/>*/}
                                                <div className='gb-input-widget'>
                                                    <div className='day-buttons'>
                                                        <div>
                                                            <a className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => setCurrentDate(addDays(currentDate, -1))}>
                                                                <i className='icon'>chevron_left</i>
                                                            </a>
                                                        </div>
                                                        <div>
                                                            <a className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' onClick={() => setCurrentDate(addDays(currentDate, 1))}>
                                                            <i className='icon'>chevron_right</i>
                                                            </a>
                                                        </div>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
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
            </div>
            <div className='sub-overlay-close' onClick={() => props.setSubOverlayHidden(true)}><i className='icon'>close</i></div>
        </div>
    )
}