/* eslint-disable @typescript-eslint/no-explicit-any */
import { mount, VueWrapper } from '@vue/test-utils';
import TimeInputField from '../../src/components/TimeInputField.vue';
import { DateTimeService } from '../../src/services/DateTimeService';
import flushPromises from 'flush-promises';


describe('Test InputField.vue:', () => {
  let wrapper: VueWrapper;
  let dateService: DateTimeService;

  beforeEach(() => {
    wrapper = mount(TimeInputField);
    dateService = wrapper.vm.$ds as DateTimeService;
  })

  it("Correct initialization:", () => {
    // initial time correctly displayed
    expect(wrapper.find("input").element.value).toBe(dateService.getTimeString());
    // correct icon
    expect(wrapper.find("div .icon").text()).toBe("schedule");
    // arrows are displayed
    expect(wrapper.find("div .gb-input-widget").exists()).toBeTruthy();
    // label displayed
    expect(wrapper.find(".label").exists()).toBeTruthy();
    // Autocomplete not rendered
    expect(wrapper.findComponent("StationAddressAutocomplete").exists()).toBeFalsy();
    // numeric input mode
    expect(wrapper.find("input").attributes("inputmode")).toBe("numeric");
  });

  it("Snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Arrows work", async () => {
    const buttons = wrapper.findAll("button");
    const input = wrapper.find('input');
    const d = dateService.dateTime;
    buttons[0].trigger('mousedown');
    buttons[0].trigger('mouseup');
    await flushPromises();
    expect(input.element.value).toBe(dateService.getTimeString(new Date(d - 3600000).valueOf()));
    buttons[1].trigger('mousedown');
    buttons[1].trigger('mouseup');
    await flushPromises();
    expect(input.element.value).toBe(dateService.getTimeString(new Date(d).valueOf()));
  })

  it("Manual input tests", async () => {
    const input = wrapper.find('input');
    // correct time
    input.setValue("11:22");
    await flushPromises();
    expect(input.element.value).toBe("11:22");
    // incorrect time
    input.setValue("25:54");
    await flushPromises();
    expect(input.element.value).toBe("11:22");
    // incorrect value
    input.setValue("7554");
    await flushPromises();
    expect(input.element.value).toBe("11:22");
  })

  it("Emit works", async () => {
    const buttons = wrapper.findAll("button");
    for(let i = 0; i < 2; i++) {
      buttons[i].trigger('mousedown');
      buttons[i].trigger('mouseup');
      await flushPromises();
      expect(wrapper.emitted().timeChanged).toHaveLength(i + 1);
    }
  })
})
