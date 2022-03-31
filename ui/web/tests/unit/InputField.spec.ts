/* eslint-disable @typescript-eslint/no-explicit-any */
import { mount, VueWrapper } from '@vue/test-utils'
import InputField from '../../src/components/InputField.vue'
import StationAddressAutocomplete from "../../src/components/StationAddressAutocomplete.vue";

describe('Test InputField.vue:', () => {
  let wrapper: VueWrapper;

  beforeEach(() => {
    wrapper = mount(InputField, {
    props: {
      labelName: "Start",
      iconType: "Train",
      showLabel: true,
      initInputText: "Fast",
      showArrows: true,
      showAutocomplete: false,
      isTimeCalendarField: false,
    }
    })
  });

  it("Correct inititalisation", () => {
    const input = wrapper.find('input');
    expect(input.element.value).toBe("Fast");
    expect(input.attributes('class')).toBe('gb-input');
    // props correctly assigned to the correct places
    expect(wrapper.find('.label').text()).toBe("Start");
    expect(wrapper.find('.icon').text()).toBe("Train");
    // arrows are rendered
    expect(wrapper.find('div .gb-input-widget').exists()).toBeTruthy();
    // showAutocomplete is false ==> component wouldn't render
    expect(wrapper.findComponent(StationAddressAutocomplete).exists()).toBeFalsy();
    // unassigned prop has correct default value
    expect(wrapper.props().tabIndex).toBe(0);
    // isFocused is false
    expect(wrapper.find('.gb-input-group').exists()).toBeTruthy()
    expect(wrapper.find('.gb-input-group .gb-input-group-selected').exists()).toBeFalsy();
  });

  it("Matches to snapshot", () => {
    expect(wrapper.html()).toMatchSnapshot();
  })

  it("Entered value recognized", () => {
    const input = wrapper.find('input');
    // value entered
    input.setValue('Test');
    // value found
    expect(input.element.value).toBe('Test');
    expect((wrapper.vm.$data as any).inputValue).toBe('Test')
    input.setValue('');
    expect(input.element.value).toBe('');
    expect((wrapper.vm.$data as any).inputValue).toBe('')
    input.setValue('315Da');
    expect(input.element.value).toBe('315Da');
    expect((wrapper.vm.$data as any).inputValue).toBe('315Da')
  });

  it("Testing emits", () => {
    const input = wrapper.find('input');
    // focus event emitted
    input.trigger('focus');
    expect(wrapper.emitted().focus).toHaveLength(1);
    // keydown event emitted
    input.trigger('keydown');
    expect(wrapper.emitted().keydown).toHaveLength(1);
    // mousedown doesn't trigger mouseup event
    input.trigger('mousedown');
    input.trigger('mouseup');
    expect(wrapper.emitted().mouseup).toHaveLength(1);
    // when input changed emits emitted and input value sended
    input.setValue('Test');
    expect(wrapper.emitted().inputChanged).toHaveLength(1);
    expect(wrapper.emitted().inputChanged[0]).toStrictEqual(["Test"]);
    expect(wrapper.emitted().inputChangedNative).toHaveLength(1);
    // another try
    input.setValue('Second');
    expect(wrapper.emitted().inputChanged).toHaveLength(2);
    expect(wrapper.emitted().inputChanged[1]).toStrictEqual(["Second"]);
    expect(wrapper.emitted().inputChangedNative).toHaveLength(2);
    // blur event emitted
    input.trigger('blur');
    expect(wrapper.emitted().blur).toHaveLength(1);
  })
})
