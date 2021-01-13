import { formatDateTime } from "./util/dateFormat";

function TimeControl(props) {
  const buttonClass = `bg-gray-200 px-2 py-1 border border-gray-300 rounded-xl ${
    props.disabled ? "text-gray-300" : ""
  }`;
  return (
    <div className="flex flex-row items-center space-x-2 m-2">
      {props.systemTime ? (
        <>
          <div>System time: {formatDateTime(props.systemTime)}</div>
          {[1, 5, 10, 30].map((min) => (
            <button
              key={`${min}m`}
              className={buttonClass}
              disabled={props.disabled}
              onClick={() => props.onForwardTime(props.systemTime + 60 * min)}
            >
              +{min}m
            </button>
          ))}
          {[1, 5, 6, 10, 12, 24].map((hrs) => (
            <button
              key={`${hrs}h`}
              className={buttonClass}
              disabled={props.disabled}
              onClick={() =>
                props.onForwardTime(props.systemTime + 60 * 60 * hrs)
              }
            >
              +{hrs}h
            </button>
          ))}
        </>
      ) : (
        <div>System time: loading...</div>
      )}
    </div>
  );
}

export default TimeControl;
