from abc import ABC, abstractmethod


class Component(ABC):
    """Declare the interface for objects in the composition.

    Implement default behavior for the interface common to all classes,
    as appropriate.

    Declare an interface for accessing and managing its child
    components.

    Define an interface for accessing a component's parent in the
    recursive structure, and implement it if that's appropriate
    (optional).
    """

    @abstractmethod
    def operation(self, *args, **kwargs):
        pass


class TgtComposite(Component):
    """Define behavior for components having children and store child components.

    Attributes:
        kwargs: The keyword arguments.
    """

    def __init__(self, **kwargs):
        self._children = set()

        self.kwargs = kwargs

    def operation(self, model, buckets):
        """The operation applied to each child component.

        Args:
            model (obj): The model.
            buckets (list): The list of buckets.

        Return:
            tgt_obj_dict (dict): The dictionary of target objects.
        """

        tgt_obj_dict = dict()
        for child_indx, child in enumerate(self._children):
            tgt_obj = child.operation(model[buckets[child_indx]], **self.kwargs)

            tgt_obj_dict[child_indx] = tgt_obj

        return tgt_obj_dict

    def execute(self, z, target, tgt_list):
        """Compute target labels for the attack.

        Args:
            z: The features.
            target: The target.
            tgt_list: The target list.

        Return:
            The dictionary of target labels
        """

        tgt_dict = dict()
        for i, tgt in enumerate(tgt_list):
            target_labels = tgt.get_target_label(z, target)

            tgt_dict[i] = target_labels

        return tgt_dict

    def add(self, component):
        """Add component to the children.

        Args:
            component: The component to be added.
        """

        self._children.add(component)

    def remove(self, component):
        """Remove component to the children.

        Args:
            component: The component to be removed.
        """

        self._children.discard(component)


class TgtLeaf(Component):
    """Represent leaf objects in the composition. A leaf has no children.

    Define behavior for primitive objects in the composition.
    """

    def operation(self, model, **kwargs):

        self.tgt_l_obj = kwargs["tgt_l"](model, **kwargs)

        return self.tgt_l_obj
