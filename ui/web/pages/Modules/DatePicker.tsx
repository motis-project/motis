import React from 'react';

import moment from 'moment';

import { Translations } from './Localization';


function useOutsideAlerter(ref: React.MutableRefObject<any>, inputFieldRef: React.MutableRefObject<any>, dayButtonPrevious: React.MutableRefObject<any>, dayButtonNext: React.MutableRefObject<any>, setShowDatePicker: React.Dispatch<React.SetStateAction<boolean>>) {
    React.useEffect(() => {
        /**
         * Alert if clicked on outside of element
         */
        function handleClickOutside(event) {
            if (ref.current && !ref.current.contains(event.target) && 
                inputFieldRef.current && !inputFieldRef.current.contains(event.target) &&
                dayButtonPrevious.current && !dayButtonPrevious.current.contains(event.target) &&
                dayButtonNext.current && !dayButtonNext.current.contains(event.target)) {
                setShowDatePicker(false);
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


export const DatePicker: React.FC<{'translation': Translations, 'currentDate': moment.Moment, 'setCurrentDate': React.Dispatch<React.SetStateAction<moment.Moment>>}> = (props) => {
    
    const datePickerRef = React.useRef(null);

    const inputFieldRef = React.useRef(null);

    const dayButtonPrevious = React.useRef(null);

    const dayButtonNext = React.useRef(null);

    const[datePickerSelected, setDatePickerSelected] = React.useState<Boolean>(false);
    
    const[currentDate, setcurrentDate] = React.useState<moment.Moment>(props.currentDate);
    
    const[dateDisplay, setDateDisplay] = React.useState<string>(currentDate.format('D.M.YYYY'));
    
    useOutsideAlerter(datePickerRef, inputFieldRef, dayButtonPrevious, dayButtonNext, setDatePickerSelected);

    React.useEffect(() => {
        setcurrentDate(props.currentDate);
    }, [props.currentDate]);

    React.useEffect(() => {
        props.setCurrentDate(currentDate.clone());
    }, [dateDisplay]);

    const weekdayshortname = props.translation.search.weekDays.map(day => {
        return (
            <th key={day.toString()} className='week-day'>
                {day}
            </th>
        );
     });

    const firstDayOfMonth = () => {
        return moment(currentDate).startOf('month').format('d') as unknown as number;
    };

    let blanks = [];
    let previousMonthDays = moment(currentDate).subtract(1, 'month').daysInMonth();
    for (let i = 0; i < firstDayOfMonth()-1; i++) {
        blanks.push(
            <td className='out-of-month' 
                key={'previous-month' + i.toString()}
                onClick={() => {
                    let firstDay = firstDayOfMonth();
                    setcurrentDate(currentDate.subtract(1, 'month'));
                    setcurrentDate(currentDate.date(currentDate.daysInMonth() - firstDay + 2 + i));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {previousMonthDays - firstDayOfMonth() + 2 + i}
            </td>
        );
    };

    let today = () => {  
        return moment().format('D') as unknown as number;
    };

    let isToday = (d: number) => {
        return moment().format('D.M.YYYY') === moment(currentDate).date(d).format('D.M.YYYY');
    }

    let selectedDay = () => {
        return currentDate.format('D') as unknown as number;
    };
        
    let daysInMonth = [];
    for (let d = 1; d <= currentDate.daysInMonth(); d++) {
        let currentDay = isToday(d) ? ' today' : '';
        let selected = d == selectedDay() ? ' selected' : '';
        let validDay = d >= today() ? ' valid-day' : ' invalid-day';
        daysInMonth.push(
            <td key={d} 
                className={`in-month${currentDay}${selected}${validDay}`}
                onClick={() => {
                    setcurrentDate(currentDate.date(d));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {d}
            </td>
        );
    };

    let fillRemainingDays = [];
    for (let i = 1; i <= 42 - [...blanks, ...daysInMonth].length; i++) {
        fillRemainingDays.push(
            <td key={'next-month' + i.toString()}
                className='out-of-month invalid-day'
                onClick={() => {
                    setcurrentDate(currentDate.add(1, 'month'));
                    setcurrentDate(currentDate.date(i));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {i}    
            </td>
        )
    }

    let totalSlots = [...blanks, ...daysInMonth, ...fillRemainingDays];
    let rows = [];
    let cells = [];

    totalSlots.forEach((row, i) => {
        if (i % 7 !== 0) {
            cells.push(row); // if index is not equal to 7 dont go to next week
        } else {
            rows.push(cells); // when we reach next week we push all td in last week to rows 
            cells = []; // empty container 
            cells.push(row); // in current loop we still push current row to new container
        }
        if (i === totalSlots.length - 1) { // when loop ends we add remain date
            rows.push(cells);
        }
    });

    let daysinmonth = rows.map((d, i) => {
        return <tr key={i}>{d}</tr>;
    });

    return (
        <div>
            <div>
                <div className='label'>{props.translation.search.date}</div>
                <div className='gb-input-group'>
                    <div className='gb-input-icon'>
                        <i className='icon'>event</i></div>
                    <input  className='gb-input' 
                            ref={inputFieldRef}
                            tabIndex={3} 
                            value={dateDisplay}
                            onChange={(e) => {
                                setDateDisplay(e.currentTarget.value);
                            }} 
                            onFocus={() => setDatePickerSelected(true)}/>
                    <div className='gb-input-widget'>
                        <div className='day-buttons'>
                            <div>
                                <a  className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                    ref={dayButtonPrevious}
                                    onClick={() => {
                                        setcurrentDate(currentDate.subtract(1, 'd')); 
                                        setDateDisplay(currentDate.format('D.M.YYYY'));
                                        }}>
                                    <i className='icon'>chevron_left</i>
                                </a>
                            </div>
                            <div>
                                <a  className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                    ref={dayButtonNext}
                                    onClick={() => {
                                        setcurrentDate(currentDate.add(1, 'd')); 
                                        setDateDisplay(currentDate.format('D.M.YYYY'));
                                        }}>
                                    <i className='icon'>chevron_right</i>
                                </a>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            <div ref={datePickerRef} className={datePickerSelected ? 'paper calendar' : 'paper calendar hide'}>
                <div className='month'>
                    <i className='icon' onClick={() => {setcurrentDate(currentDate.subtract(1, 'month')); setDateDisplay(currentDate.format('D.M.YYYY'))}}>chevron_left</i>
                    <span className='month-name'>{props.translation.search.months[currentDate.month()] + ' ' + currentDate.year()}</span>
                    <i className='icon' onClick={() => {setcurrentDate(currentDate.add(1, 'month')); setDateDisplay(currentDate.format('D.M.YYYY'))}}>chevron_right</i>
                </div>
                <table className='calendar-day'>
                    <thead className='weekdays'>
                        <tr>{weekdayshortname}</tr> 
                    </thead>
                    <tbody className='calendardays'>{daysinmonth}</tbody>
                </table>
            </div>
        </div>
    )
}