import React from 'react';

import moment from 'moment';

import { Translations } from '../App/Localization';
import { Interval } from '../Types/RoutingTypes';


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


// Check if day is in the interval.
const isValidDay = (day: moment.Moment, interval: Interval) => {
    if (interval == null) {
        return '';
    } else {
        if (day.unix() >= interval.begin && day.unix() < interval.end) {
            //console.log(day.unix(), interval.begin, interval.end)
            return 'valid-day';
        }else {
            return 'invalid-day';
        };
    }
}


export const DatePicker: React.FC<{'translation': Translations, 'currentDate': moment.Moment, 'setCurrentDate': React.Dispatch<React.SetStateAction<moment.Moment>>, 'scheduleInfo': Interval}> = (props) => {
    
    const datePickerRef = React.useRef(null);

    const inputFieldRef = React.useRef(null);

    const dayButtonPrevious = React.useRef(null);

    const dayButtonNext = React.useRef(null);

    // Boolean used to decide if the datepicker is visible or not
    const[datePickerSelected, setDatePickerSelected] = React.useState<Boolean>(false);
    
    // Holds the currently displayed Date as moment.Moment Object
    const[currentDate, setCurrentDate] = React.useState<moment.Moment>(moment());
    
    // Holds the currently displayed Date as String. This additional State is needed to handle the onChange Event for custom Input
    const[dateDisplay, setDateDisplay] = React.useState<string>(null);
    
    useOutsideAlerter(datePickerRef, inputFieldRef, dayButtonPrevious, dayButtonNext, setDatePickerSelected);

    React.useEffect(() => {
        if (props.currentDate) {
            setCurrentDate(props.currentDate);
            setDateDisplay(props.currentDate.format('D.M.YYYY'))
        }
    }, [props.currentDate]);

    /*React.useEffect(() => {
        props.setCurrentDate(currentDate.clone());
    }, [dateDisplay]);*/
    
    // Create weekday name elements.
    const weekdayshortname = props.translation.search.weekDays.map(day => {
        return (
            <th key={day.toString()} className='week-day'>
                {day}
            </th>
        );
     });
    
    // Get weekday number for first Day in current month.
    const firstDayOfMonth = () => {
        let fd = moment(currentDate).startOf('month').format('d') as unknown as number;
        return fd == 0 ? 7 : fd;
    };
    
        // Used for setting className of td correctly. Returns 'today' if this td is representing today.
        let isToday = (d: number, date: moment.Moment) => {
            return moment().format('D.M.YYYY') === moment(date).date(d).format('D.M.YYYY') ? 'today ' : '';
        }
    
        // Used for setting className of td correctly. Returns 'selected' if this td is currently displayed in the search.
        let selectedDay = (d: number) => {
            return currentDate.format('D') as unknown as number == d ? 'selected ' : '';
        };

    // daysInPreviousMonth contains all days of the previous Month that have to be shown if the 1. doesnt fall on a monday.
    let daysInPreviousMonth = [];
    let dayToAdd = moment(currentDate).utc().startOf('month').subtract(firstDayOfMonth(), 'd');
    for (let d = 0; d < firstDayOfMonth()-1; d++) {
        dayToAdd.add(1, 'd');
        daysInPreviousMonth.push(
            <td className={`out-of-month ${isToday(d, dayToAdd)}${isValidDay(dayToAdd, props.scheduleInfo)}`} 
                key={'previous-month' + d.toString()}
                onClick={() => {
                    let firstDay = firstDayOfMonth();
                    setCurrentDate(currentDate.subtract(1, 'month'));
                    setCurrentDate(currentDate.date(currentDate.daysInMonth() - firstDay + 2 + d));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {dayToAdd.date()}
            </td>
        );
    };
    
    // daysInMonth contains all days of this Month.
    let daysInMonth = [];
    dayToAdd = moment(currentDate).utc().startOf('month').subtract(1, 'd');
    for (let d = 1; d <= currentDate.daysInMonth(); d++) {
        dayToAdd.add(1, 'd');
        daysInMonth.push(
            <td key={d} 
                className={`in-month ${isToday(d, dayToAdd)}${selectedDay(d)}${isValidDay(dayToAdd, props.scheduleInfo)}`}
                onClick={() => {
                    setCurrentDate(currentDate.date(d));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {d}
            </td>
        );
    };

    // fillRemainingDays contains all days of the next month that are needed to display 42 Days in total.
    let fillRemainingDays = [];
    dayToAdd = moment(currentDate).utc().startOf('month').add(1, 'M').subtract(1, 'd');
    for (let d = 1; d <= 42 - [...daysInPreviousMonth, ...daysInMonth].length; d++) {
        dayToAdd.add(1, 'd');
        fillRemainingDays.push(
            <td key={'next-month' + d.toString()}
                className={`out-of-month ${isToday(d, dayToAdd)}${isValidDay(dayToAdd, props.scheduleInfo)}`}
                onClick={() => {
                    setCurrentDate(currentDate.add(1, 'month'));
                    setCurrentDate(currentDate.date(d));
                    setDateDisplay(currentDate.format('D.M.YYYY'));
                    setDatePickerSelected(false);
                }}>
                {d}    
            </td>
        )
    }

    // Combine all 3 arrays into 1 total array. For every element an entry in the datepicker will be created
    let totalSlots = [...daysInPreviousMonth, ...daysInMonth, ...fillRemainingDays];
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
                            value={dateDisplay ? dateDisplay : ''}
                            onChange={(e) => {
                                setDateDisplay(e.currentTarget.value);
                                if (e.currentTarget.value.split('.').length == 3) {
                                    let [day, month, year] = e.currentTarget.value.split('.');
                                    if (day !== '' && !isNaN(+day) && month !== '' && !isNaN(+month) && year !== '' && !isNaN(+year)){
                                        let newDate = moment(currentDate);
                                        newDate.year(year as unknown as number);
                                        newDate.month(month as unknown as number - 1);
                                        newDate.date(day as unknown as number);
                                        setCurrentDate(newDate);
                                    }
                                }
                            }}
                            onKeyDown={(e) => {
                                if (e.key == 'Enter'){
                                    //console.log(searchDate)
                                    setDateDisplay(currentDate.format('D.M.YYYY'));
                                }
                            }}
                            onFocus={() => setDatePickerSelected(true)}/>
                    <div className='gb-input-widget'>
                        <div className='day-buttons'>
                            <div>
                                <a  className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                    ref={dayButtonPrevious}
                                    onClick={() => {
                                        setCurrentDate(currentDate.subtract(1, 'd')); 
                                        setDateDisplay(currentDate.format('D.M.YYYY'));
                                        }}>
                                    <i className='icon'>chevron_left</i>
                                </a>
                            </div>
                            <div>
                                <a  className='gb-button gb-button-small gb-button-circle gb-button-outline gb-button-PRIMARY_COLOR disable-select' 
                                    ref={dayButtonNext}
                                    onClick={() => {
                                        setCurrentDate(currentDate.add(1, 'd')); 
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
                    <i className='icon' onClick={() => {setCurrentDate(currentDate.subtract(1, 'month')); setDateDisplay(currentDate.format('D.M.YYYY'))}}>chevron_left</i>
                    <span className='month-name'>{currentDate ? props.translation.search.months[currentDate.month()] + ' ' + currentDate.year() : ''}</span>
                    <i className='icon' onClick={() => {setCurrentDate(currentDate.add(1, 'month')); setDateDisplay(currentDate.format('D.M.YYYY'))}}>chevron_right</i>
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