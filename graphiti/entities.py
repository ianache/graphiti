from pydantic import BaseModel, Field
from typing import Any, TypedDict, cast

class Requirement(BaseModel):
    """Represents a specific need or functionality that a system must fulfill.

    This class is used to model requirements, which are essential for defining
    the scope and behavior of a project. Each requirement is associated with a
    specific project and includes a detailed description.

    Attributes:
        project_name: The name of the project to which the requirement belongs.
        description: A detailed description of the requirement.
    """

    project_name: str = Field(
        ...,
        description="The name of the project to which the requirement belongs.",
    )
    description: str = Field(
        ...,
        description="Description of the requirement. Only use information mentioned in the context to write this description.",
    )

class Preference(BaseModel):
    """Represents a user's expressed preference for a particular item or concept.

    This class is used to model user preferences, which can be categorized
    to provide more personalized experiences. Each preference includes a
    category and a detailed description.

    Attributes:
        category: The category of the preference (e.g., 'Food', 'Music').
        description: A detailed description of the preference.
    """

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description="Brief description of the preference. Only use information mentioned in the context to write this description.",
    )


class Procedure(BaseModel):
    """Represents a set of actions to be performed in a specific scenario.

    This class is used to model procedures, which consist of a series of
    steps or instructions. Each procedure is defined by a detailed description
    that outlines the actions to be taken.

    Attributes:
        description: A detailed description of the procedure.
    """

    description: str = Field(
        ...,
        description="Brief description of the procedure. Only use information mentioned in the context to write this description.",
    )
