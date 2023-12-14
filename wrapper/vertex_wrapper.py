# Import dependencies.
from langchain.callbacks.base import BaseCallbackHandler
from langchain.schema import AgentAction, AgentFinish, Document, LLMResult
from prettyprinter import cpprint
from typing import Any, Dict, List, Optional, Sequence, Type, Union
from uuid import UUID

class Color:
    """For easier understanding and faster manipulation of printed colors."""

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    ITALICS = "\x1B[3m"
    END = "\033[0m\x1B[0m"


class OutputFormatter:
    """Helper class to control the format of printed output from the callbacks.

    If used in prod, consider reimplementing in a way that removes hardcoding
      of where the output is written. Maybe use Python logging and then pass a
      custom configuration?
    """

    def heading(text: str) -> None:
        print(f"{Color.BOLD}{text}{Color.END}")

    def key_info(text: str) -> None:
        print(f"{Color.BOLD}{Color.DARKCYAN}{text}{Color.END}")

    def key_info_labeled(
        label: str, contents: str, contents_newlined: Optional[bool] = False
    ) -> None:
        print(
            f"{Color.BOLD}{Color.DARKCYAN}{label}: {Color.END}{Color.DARKCYAN}", end=""
        )
        if contents_newlined:
            contents = contents.splitlines()
        cpprint(f"{contents}")
        print(f"{Color.END}", end="")

    def debug_info(text: str) -> None:
        print(f"{Color.BLUE}{text}{Color.END}")

    def debug_info_labeled(
        label: str, contents: str, contents_newlined: Optional[bool] = False
    ) -> None:
        print(f"{Color.BOLD}{Color.BLUE}{label}: {Color.END}{Color.BLUE}", end="")
        if contents_newlined:
            contents = contents.splitlines()
        cpprint(f"{contents}")
        print(f"{Color.END}", end="")

    def llm_call(text: str) -> None:
        print(f"{Color.ITALICS}{text}{Color.END}")

    def llm_output(text: str) -> None:
        print(f"{Color.UNDERLINE}{text}{Color.END}")

    def tool_call(text: str) -> None:
        print(f"{Color.ITALICS}{Color.PURPLE}{text}{Color.END}")

    def tool_output(text: str) -> None:
        print(f"{Color.UNDERLINE}{Color.PURPLE}{text}{Color.END}")

    def debug_error(text: str) -> None:
        print(f"{Color.BOLD}{Color.RED}{text}{Color.END}")


# Actual Langchain callback handler, this produces status updates during a
#   Langchain execution.
class AllChainDetails(BaseCallbackHandler):
    """Outputs details of chain progress and state.

    Exposes details available at callback time to each executed step in a chain.

    Method arguments in this class are based on the (most of?) the arguments
      available to the callback method, though not all implementations in this
      class use all the arguments.

    Usage:
      Pass as an argument to a langchain method or class that accepts a callback
        handler. Note that  not all langchain classes will invoke all callbacks
        when the callback handler is provided at initialization time, so the
        recommended usage is to provide the callback handler when executing a
        chain.

    Example:
      from langchain import LLMChain, PromptTemplate
      from langchain.llms import VertexAI
      import vertexai  # Comes from google-cloud-aiplatform package.
      vertexai.init(project=PROJECT_ID, location=REGION)

      llm = VertexAI(temperature=0)  # Use any LLM.
      prompt_template = "What food pairs well with {food}?"
      handler = AllChainDetails()
      llm_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate.from_template(prompt_template))
      llm_chain("chocolate", callbacks=[handler])

    Args:
      debug_mode: If True, prints more details of each chain step and activates
        breakpoints (using pdb) when unexpected behavior is detected. Note that
        the breakpoints are in the callbacks, which limits the amount of
        inspectable langchain state to what langchain surfaces to callbacks.
      out: Class for managing output, only tested with the OutputFormatter
        accompanying this class.
    """

    def __init__(
        self,
        debug_mode: Optional[bool] = False,
        out: Type[OutputFormatter] = OutputFormatter,
    ) -> None:
        self.debug_mode = debug_mode
        self.out = out

    def on_text(
        self,
        text: str,
        color: Optional[str] = None,
        end: str = "",
        **kwargs: Any,
    ) -> None:
        """Run usually (not always) when langchain creates text for an LLM call.

        This callback is only used when debug_mode == True, since it can be
          confusing to see the blocks of text that come from this callback on top
          of the text sent to the LLM--it's much easier to understand what's going
          on by only looking at text sent to an LLM.

        """
        if self.debug_mode:
            self.out.heading(f"\n\n> Preparing text.")
            self.out.debug_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
            self.out.debug_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            print(text)  # Langchain already agressively formats this.

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when langchain calls an LLM."""
        self.out.heading(f"\n\n> Sending text to the LLM.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if len(prompts) > 1:
            self.out.debug_error("prompts has multiple items.")
            self.out.debug_error("Only outputting first item in prompts.")
            if self.debug_mode:
                self.out.debug_info_labeled("Prompts", f"{prompts}")
                breakpoint()

        self.out.key_info(f"Text sent to LLM:")
        self.out.llm_call(prompts[0])

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("serialized", f"{serialized}")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Run after LLM response is received by langchain."""
        self.out.heading(f"\n\n> Received response from LLM.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if len(response.generations) > 1:
            self.out.debug_error("response object has multiple generations.")
            self.out.debug_error("Only outputting first generation in response.")
            if self.debug_mode:
                self.out.debug_info_labeled("response", f"{response}")
                breakpoint()

        self.out.key_info(f"Text received from LLM:")
        self.out.llm_output(response.generations[0][0].text)

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("response", f"{response}")

    def on_chain_start(
        self, serialized: Dict[str, Any], inputs: Dict[str, Any], **kwargs: Any
    ) -> None:
        """Run when a new chain (or subchain) is started."""
        self.out.heading(f"\n\n> Starting new chain.")

        if "id" not in serialized.keys():
            self.out.debug_error("Missing serialized['id']")
            class_name = "Unknown -- serialized['id'] is missing"
            if self.debug_mode:
                self.out.debug_info_labeled("serialized", f"{serialized}")
                breakpoint()
        else:
            class_name = ".".join(serialized["id"])

        self.out.key_info_labeled(f"Chain class", f"{class_name}")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if len(inputs) < 1:
            self.out.debug.error("Chain inputs is empty.")
            if self.debug_mode:
                self.out.debug_info_labeled("inputs", f"{inputs}")
                breakpoint()
        else:
            self.out.key_info("Iterating through keys/values of chain inputs:")
        for key, value in inputs.items():
            # These keys contain mostly noise.
            if key not in ["stop", "agent_scratchpad"]:
                self.out.key_info_labeled(f"   {key}", f"{value}")

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("inputs", f"{inputs}")
            self.out.debug_info_labeled("serialized", f"{serialized}")

    def on_chain_end(self, outputs: Dict[str, Any], **kwargs: Any) -> None:
        """Run when a chain completes."""
        self.out.heading(f"\n\n> Ending chain.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if len(outputs) == 0:
            self.out.debug_errors("No chain outputs.")
            if self.debug_mode:
                self.out.debug_info_labeled("outputs", f"{outputs}")
                breakpoint()
        else:
            outputs_keys = [*outputs.keys()]
        for key in outputs_keys:
            self.out.key_info_labeled(
                f"Output {key}", f"{outputs[key]}", contents_newlined=True
            )

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("outputs", f"{outputs}")

    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        **kwargs: Any,
    ) -> None:
        """Run when making a call to a tool."""
        self.out.heading(f"\n\n> Using tool.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")
        self.out.key_info_labeled(f"Tool name", f"{serialized['name']}")
        self.out.key_info(f"Query sent to tool:")
        self.out.tool_call(input_str)

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("serialized", f"{serialized}")

    def on_tool_end(
        self,
        output: str,
        color: Optional[str] = None,
        observation_prefix: Optional[str] = None,
        llm_prefix: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Run on response from a tool."""
        self.out.heading(f"\n\n> Received tool output.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")
        self.out.key_info_labeled(f"Tool name", f"{kwargs['name']}")

        if "output" not in locals():
            self.out.debug_error("No tool output.")
            if self.debug_mode:
                breakpoint()
        else:
            self.out.key_info("Response from tool:")
            self.out.tool_output(f"{output}")

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("observation_prefix", f"{observation_prefix}")
            self.out.debug_info_labeled("llm_prefix", f"{llm_prefix}")

    def on_agent_action(
        self, action: AgentAction, color: Optional[str] = None, **kwargs: Any
    ) -> Any:
        """Run when agent performs an action."""
        self.out.heading(f"\n\n> Agent taking an action.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if not hasattr(action, "log"):
            self.out.debug_error("No log in action.")
            if self.debug_mode:
                self.out.debug_info_labeled("action", f"{action}")
                breakpoint()
        else:
            self.out.key_info_labeled(
                f"Action log", f"{action.log}", contents_newlined=True
            )

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("action", f"{action}")

    def on_agent_finish(
        self, finish: AgentFinish, color: Optional[str] = None, **kwargs: Any
    ) -> None:
        """Run after agent completes."""
        self.out.heading(f"\n\n> Agent has finished.")
        self.out.key_info_labeled(f"Chain ID", f"{kwargs['run_id']}")
        self.out.key_info_labeled("Parent chain ID", f"{kwargs['parent_run_id']}")

        if not hasattr(finish, "log"):
            self.out.debug_error("No log in action finish.")
            if self.debug_mode:
                breakpoint()
        else:
            self.out.key_info_labeled(
                f"Action finish log", f"{finish.log}", contents_newlined=True
            )

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("finish", f"{finish}")

    def on_llm_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.out.debug_error("LLM Error")
        self.out.debug_info_labeled("Error object", f"{error}")
        if self.debug_mode:
            breakpoint()

    def on_chain_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.out.debug_error("Chain Error")
        self.out.debug_info_labeled("Error object", f"{error}")
        if self.debug_mode:
            breakpoint()

    def on_tool_error(
        self, error: Union[Exception, KeyboardInterrupt], **kwargs: Any
    ) -> None:
        self.out.debug_error("Chain Error")
        self.out.debug_info_labeled("Error object", f"{error}")
        if self.debug_mode:
            breakpoint()

    def on_retriever_start(
        self,
        serialized: Dict[str, Any],
        query: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when querying a retriever."""
        self.out.heading(f"\n\n> Querying retriever.")
        self.out.key_info_labeled(f"Chain ID", f"{run_id}")
        self.out.key_info_labeled("Parent chain ID", f"{parent_run_id}")
        self.out.key_info_labeled("Tags", f"{tags}")

        if "id" not in serialized.keys():
            self.out.debug_error("Missing serialized['id']")
            class_name = "Unknown -- serialized['id'] is missing"
            if self.debug_mode:
                self.out.debug_info_labeled("serialized", f"{serialized}")
                breakpoint()
        else:
            class_name = ".".join(serialized["id"])
        self.out.key_info_labeled(f"Retriever class", f"{class_name}")

        self.out.key_info(f"Query sent to retriever:")
        self.out.tool_call(query)

        if self.debug_mode:
            self.out.debug_info_labeled("Arguments", f"{kwargs}")
            self.out.debug_info_labeled("metadata", f"{metadata}")
            self.out.debug_info_labeled("serialized", f"{serialized}")

    def on_retriever_end(
        self,
        documents: Sequence[Document],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs: Any,
    ) -> Any:
        """Run when retriever returns a response."""
        self.out.heading(f"\n\n> Retriever finished.")
        self.out.key_info_labeled(f"Chain ID", f"{run_id}")
        self.out.key_info_labeled("Parent chain ID", f"{parent_run_id}")
        self.out.key_info(f"Found {len(documents)} documents.")

        if len(documents) == 0:
            self.out.debug_error("No documents found.")
            if self.debug_mode:
                breakpoint()
        else:
            for doc_num, doc in enumerate(documents):
                self.out.key_info("---------------------------------------------------")
                self.out.key_info(f"Document number {doc_num} of {len(documents)}")
                self.out.key_info_labeled("Metadata", f"{doc.metadata}")
                self.out.key_info("Document contents:")
                self.out.tool_output(doc.page_content)
