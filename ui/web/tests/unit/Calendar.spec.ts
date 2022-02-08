/* eslint-disable @typescript-eslint/no-explicit-any */
import { mount, VueWrapper } from '@vue/test-utils'
import flushPromises from 'flush-promises';
import Calendar from '../../src/components/Calendar.vue'
import { DateTimeService } from '../../src/services/DateTimeService'
import { getRandomInt } from "./TestHelpers"

describe('Test Calendar.vue', () => {
  let wrapper: VueWrapper;
  let dateService: DateTimeService;

  beforeEach(() => {
    wrapper = mount(Calendar)
    dateService = wrapper.vm.$ds as DateTimeService;
  });

  it("Snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it('Renders input field', () => {
    const inputField = wrapper.getComponent({ name: "InputField" })
    expect(inputField.get("input").element.value).toBe(dateService.getDateString());
  });

  it('Shows and hides calendar', async () => {
    const calendar = wrapper.get("div .calendar");
    expect(calendar.isVisible()).toBeFalsy();
    wrapper.get("input").trigger("focus");
    await flushPromises();
    expect(calendar.isVisible()).toBeTruthy();
    wrapper.get("input").trigger("blur");
    await flushPromises();
    expect(calendar.isVisible()).toBeFalsy();
  });

  it("Day change with buttons", async () => {
    const inputField = wrapper.getComponent({ name: "InputField" })
    const buttons = inputField.findAll(".gb-button");
    expect(buttons).toHaveLength(2);
    const date: Date = dateService.date;
    const dates: Date[] = [new Date(date.getFullYear(), date.getMonth(), date.getDate() - 1,
      date.getHours(), date.getMinutes(), date.getSeconds(), date.getMilliseconds()), date]
    for(let i = 0; i < 2; i++) {
      buttons[i].trigger("mousedown");
      await flushPromises();
      buttons[i].trigger("mouseup");
      await flushPromises();
      expect(inputField.get("input").element.value).toBe(dateService.getDateString(dates[i].valueOf()));
    }
  });

  it("Month change with buttons", async () => {
    const inputField = wrapper.getComponent({ name: "InputField" })
    const calendar = wrapper.get("div .calendar");
    const buttons = wrapper.get(".month").findAll("i");
    expect(buttons).toHaveLength(2);
    const date: Date = dateService.date;
    const dates: Date[] = [new Date(date.getFullYear(), date.getMonth() - 1, date.getDate(),
      date.getHours(), date.getMinutes(), date.getSeconds(), date.getMilliseconds()), date];
    inputField.get("input").trigger("focus");
    await flushPromises();
    expect(calendar.isVisible()).toBeTruthy();
    for(let i = 0; i < 2; i++) {
      buttons[i].trigger("click");
      await flushPromises();
      expect(inputField.get("input").element.value).toBe(dateService.getDateString(dates[i].valueOf()));
    }
    expect(calendar.isVisible()).toBeTruthy();
  });

  it('Day in calendar clicked', async () => {
    const inputField = wrapper.getComponent({ name: "InputField" })
    const calendar = wrapper.get("div .calendar");
    const randomDayString = getRandomInt(1, 28).toString();
    const day = wrapper.findAll(".calendardays li").filter(node => node.text() === randomDayString)[0];
    inputField.get("input").trigger("focus");
    await flushPromises();
    expect(calendar.isVisible()).toBeTruthy();
    day.trigger("mousedown");
    await flushPromises();
    expect(day.text()).toBe(randomDayString);
    expect(inputField.get("input").element.value).toBe(dateService.getDateString((wrapper.vm.$data as any).currentDate.valueOf()))
    expect((wrapper.vm.$data as any).currentDate.getDate()).toBe(Number.parseInt(randomDayString));
    expect(calendar.isVisible()).toBeFalsy();
  });

  it('Valid input in field', async () => {
    expect(wrapper.emitted('dateChanged')).toHaveLength(1);
    const inputField = wrapper.getComponent({ name: "InputField" }).get("input");
    const date = dateService.date;
    const newDate: Date = new Date(getRandomInt(2000, 2020), getRandomInt(0, 11), getRandomInt(1, 28),
      date.getHours(), date.getMinutes(), date.getSeconds(), date.getMilliseconds());
    inputField.setValue(dateService.getDateString(newDate.valueOf()));
    await flushPromises();
    const dateChanged = wrapper.emitted('dateChanged');
    expect(dateChanged).toHaveLength(2);
    expect(dateChanged !== undefined ? (dateChanged[1] as Date[])[0].valueOf() : null).toBe(newDate.valueOf())
  });

  it('Invalid input in field', () => {
    expect(wrapper.emitted('dateChanged')).toHaveLength(1);
    const inputField = wrapper.getComponent({ name: "InputField" }).get("input");
    const date = dateService.date;
    const newDate: Date = new Date(getRandomInt(2000, 2020), getRandomInt(0, 11), getRandomInt(1, 28),
      date.getHours(), date.getMinutes(), date.getSeconds(), date.getMilliseconds());
    inputField.setValue(dateService.getDateString(newDate.valueOf()) + "invalid string");
    const dateChanged = wrapper.emitted('dateChanged');
    expect(dateChanged).toHaveLength(1);
    inputField.trigger("blur");
  });
})
