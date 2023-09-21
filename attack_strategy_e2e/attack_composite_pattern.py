from abc import ABC, abstractmethod


class Component(ABC):
    """Declare the interface for objects in the composition.

    Methods to be implemented:
        - Implement default behavior for the interface common to all classes,
        as appropriate.

        - Declare an interface for accessing and managing its child
        components.

        - Define an interface for accessing a component's parent in the
        recursive structure, and implement it if that's appropriate
        (optional).
    """

    @abstractmethod
    def operation(self, *args, **kwargs):
        pass


class AttkComposite(Component):
    """Define behavior for components having children.

    Attributes:
        kwargs: The keyword arguments.
    """

    def __init__(self, **kwargs):

        self._children = set()

        self.kwargs = kwargs

    def operation(self, tgt_obj, encoder_bnchmrk, model_main, model, buckets, **kwargs):
        """Implement child-related operation in the Component interface.

        Args:
            tgt_obj (obj): The target object.
            encoder_bnchmrk (dict): The dictionary of models.
            model_main (dict): The dictionary of models.
            model (dict): The dictionary of models.
            buckets (list(int)): The list of bucket indices.

        Return:
            attk_obj_dict (dict): The dictionary of attacks.
        """

        attk_obj_dict = dict()
        for child_indx, child in enumerate(self._children):
            attk_obj = child.operation(
                tgt_obj[child_indx],
                encoder_bnchmrk[buckets[child_indx]],
                model_main[buckets[child_indx]],
                model[buckets[child_indx]],
                eps=kwargs["eps"][child_indx],
                t_main=kwargs["t_main"][child_indx],
                c=kwargs["c"][child_indx],
                kappa=kwargs["kappa"][child_indx],
                num_steps=kwargs["num_steps"][child_indx],
                lr=kwargs["lr"][child_indx],
                **self.kwargs,
            )

            attk_obj_dict[child_indx] = attk_obj

        return attk_obj_dict

    def execute(self, attk, **kwargs):
        """The forward path for the attack.

        Args:
            attk: The attack.
            kwargs: They keyword arguments.

        Return:
            z_adv: The adversarial features.
            z: The non-adversarial features.
        """

        z_adv, z = attk.forward(**kwargs)

        return z_adv, z

    def add(self, component):
        """Add the component."""
        self._children.add(component)

    def remove(self, component):
        """Remove the component."""
        self._children.discard(component)


class DoubleAttkComposite(Component):
    """Define behavior for components having children.

    Attributes:
        kwargs: The keyword arguments.
    """

    def __init__(self, **kwargs):

        self._children = set()

        self.kwargs = kwargs

    def operation(
        self,
        tgt_obj,
        encoder_bnchmrk,
        model_main,
        model,
        model_adv,
        buckets,
        **kwargs,
    ):
        """Implement child-related operation in the Component interface.

        Args:
            tgt_obj (obj): The target object.
            encoder_bnchmrk (dict): The dictionary of models.
            model_main (dict): The dictionary of models.
            model (dict): The dictionary of models.
            model_adv (dict): The dictionary of models.
            buckets (list(int)): The list of bucket indices.

        Return:
            attk_obj_dict (dict): The dictionary of attacks.
        """

        attk_obj_dict = dict()
        for child_indx, child in enumerate(self._children):
            attk_obj = child.operation(
                tgt_obj[child_indx],
                encoder_bnchmrk[buckets[child_indx]],
                model_main[buckets[child_indx]],
                model[buckets[child_indx]],
                model_adv[buckets[child_indx]],
                eps=kwargs["eps"][child_indx],
                t_main=kwargs["t_main"][child_indx],
                c=kwargs["c"][child_indx],
                kappa=kwargs["kappa"][child_indx],
                num_steps=kwargs["num_steps"][child_indx],
                lr=kwargs["lr"][child_indx],
                **self.kwargs,
            )

            attk_obj_dict[child_indx] = attk_obj

        return attk_obj_dict

    def execute(self, attk, **kwargs):
        """The forward path for the attack.

        Args:
            attk: The attack.
            kwargs: They keyword arguments.

        Return:
            z_adv: The adversarial features.
            z: The non-adversarial features.
        """

        z_adv, z = attk.forward(**kwargs)

        return z_adv, z

    def add(self, component):
        """Add the component."""
        self._children.add(component)

    def remove(self, component):
        """Remove the component."""
        self._children.discard(component)


class AttkLeaf(Component):
    """Represent leaf objects in the composition.

    Note: A leaf has no children.
    """

    def operation(self, tgt_obj, encoder_bnchmrk, model_main, model, **kwargs):
        """Implement the operation in the Component interface.

        Args:
            tgt_obj (obj): The target object.
            encoder_bnchmrk (dict): The dictionary of models.
            model_main (dict): The dictionary of models.
            model (dict): The dictionary of models.
            kwargs: The keyword arguments.

        Return:
            attk_obj (obj): The attack object.
        """

        self.attk_obj = kwargs["attk"](
            encoder_bnchmrk=encoder_bnchmrk,
            model_main=model_main,
            model=model,
            target_strategy=tgt_obj,
            **kwargs,
        )

        return self.attk_obj


class DoubleAttkLeaf(Component):
    """Represent leaf objects in the composition.

    Note: A leaf has no children.
    """

    def operation(
        self,
        tgt_obj,
        encoder_bnchmrk,
        model_main,
        model,
        model_adv,
        **kwargs,
    ):
        """Implement the operation in the Component interface.

        Args:
            tgt_obj (obj): The target object.
            encoder_bnchmrk (dict): The dictionary of models.
            model_main (dict): The dictionary of models.
            model (dict): The dictionary of models.
            model_adv (dict): The dictionary of models.
            kwargs: The keyword arguments.

        Return:
            attk_obj (obj): The attack object.
        """

        self.attk_obj = kwargs["attk"](
            encoder_bnchmrk=encoder_bnchmrk,
            model_main=model_main,
            model=model,
            model_adv=model_adv,
            target_strategy=tgt_obj,
            **kwargs,
        )

        return self.attk_obj