import { useAtom } from "jotai";

import { selectedGroupAtom } from "@/data/selectedGroup";

import classNames from "@/util/classNames";

import GroupDetails from "@/components/groups/GroupDetails";
import GroupList from "@/components/groups/GroupList";

function SelectedGroupDetails(): JSX.Element {
  const [selectedGroup] = useAtom(selectedGroupAtom);

  if (selectedGroup !== undefined) {
    return <GroupDetails groupId={selectedGroup} key={selectedGroup} />;
  } else {
    return <></>;
  }
}

type GroupsMainSectionProps = {
  visible?: boolean;
};

function GroupsMainSection({
  visible = true,
}: GroupsMainSectionProps): JSX.Element {
  const visibilityClass = visible ? "block" : "hidden";
  return (
    <>
      <div
        className={classNames(
          visibilityClass,
          "bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0"
        )}
      >
        <GroupList />
      </div>
      <div className={classNames(visibilityClass, "overflow-y-auto grow p-2")}>
        <SelectedGroupDetails />
      </div>
    </>
  );
}

export default GroupsMainSection;
