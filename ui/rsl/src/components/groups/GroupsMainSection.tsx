import { useAtom } from "jotai";

import { selectedGroupAtom } from "@/data/selectedGroup";

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

function GroupsMainSection(): JSX.Element {
  return (
    <>
      <div className="bg-db-cool-gray-200 dark:bg-gray-800 w-[25rem] overflow-y-auto p-2 shrink-0">
        <GroupList />
      </div>
      <div className="overflow-y-auto grow p-2">
        <SelectedGroupDetails />
      </div>
    </>
  );
}

export default GroupsMainSection;
