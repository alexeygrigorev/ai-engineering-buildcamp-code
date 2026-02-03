from pydantic import BaseModel, Field
from typing import Literal, Union


class PageBlock(BaseModel):
    type: str = Field(..., description="Discriminator that identifies which kind of page block this is.")


class SectionHeadingBlock(PageBlock):
    type: Literal["section_heading"] = "section_heading"
    title: str = Field(..., description="The section heading text.")


class TextBlock(PageBlock):
    type: Literal["text"] = "text"
    text: str = Field(..., description="Explanatory prose from the textbook.")


class EquationBlock(PageBlock):
    """
    A mathematical expression written in LaTeX.
    """
    type: Literal["equation"] = "equation"
    latex: str = Field(..., description="The equation in LaTeX format.")
    description: str | None = Field(
        None,
        description="Optional plain-language meaning or interpretation of the equation."
    )


class FigureBlock(PageBlock):
    type: Literal["figure"] = "figure"
    caption: str | None = Field(None, description="Figure caption or label, if present.")
    description: str = Field(
        ...,
        description="Conceptual description of what the figure shows and why it matters."
    )
    figure_number: int = Field(
        ...,
        description="Figure number as mentioned in the book."
    )


class TableBlock(PageBlock):
    type: Literal["table"] = "table"
    caption: str | None = Field(None, description="Table caption or label, if present.")
    columns: list[str] = Field(..., description="Column headers in reading order.")
    rows: list[list[str]] = Field(..., description="Table rows aligned with columns.")


PageBlockUnion = Union[
    SectionHeadingBlock,
    TextBlock,
    EquationBlock,
    FigureBlock,
    TableBlock,
]


class Page(BaseModel):
    page_number: int = Field(..., description="Printed page number in the textbook.")
    header: str | None = Field(None, description="Running page header text, if any.")
    blocks: list[PageBlockUnion] = Field(
        ...,
        description="Ordered list of extracted page blocks."
    )

    def print(self):
        print(self.page_number)
        print(self.header)

        for block in self.blocks:
            if block.type == 'text':
                print(block.text)
        
            elif block.type == 'equation':
                print(f'$${block.latex}$$')
        
            elif block.type == 'figure':
                # print(block)
                print(block.caption)
                print(block.description)
                print('Fig.', block.figure_number)
        
            else:
                print(block)
        
            print()


class PageResponse(BaseModel):
    page: Page
    cost: float